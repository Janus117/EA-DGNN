from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, PairTensor, OptTensor


class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """

    def __init__(self, emb_size):
        super(TimeEncode, self).__init__()
        self.dim = emb_size
        self.out_channels = emb_size
        self.w = nn.Linear(1, emb_size)
        self.reset_parameters()

    def reset_parameters(
        self,
    ):
        self.w.weight = nn.Parameter(
            (
                torch.from_numpy(
                    1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32)
                )
            ).reshape(self.dim, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.silu = nn.SiLU()
        self.w = nn.Linear(in_channels, hidden_channels, bias=False)
        self.v = nn.Linear(in_channels, hidden_channels, bias=False)
        self.o = nn.Linear(hidden_channels, out_channels, bias=False)

    def forward(self, o) -> torch.Tensor:
        return self.o(self.silu(self.w(o)) * self.v(o))  # swiglu


class FilterLayer(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int):
        super(FilterLayer, self).__init__()

        self.seq_len = seq_len

        self.complex_weight = nn.Parameter(
            torch.randn(1, self.seq_len // 2 + 1, hidden_dim, 2, dtype=torch.float32)
        )

        self.Dropout = nn.Dropout(0.1)

        # self.LayerNorm = nn.LayerNorm(hidden_dim)

    def forward(self, input_tensor: torch.Tensor):
        batch, seq_len, hidden = input_tensor.shape

        hidden_states = input_tensor

        x = torch.fft.rfft(hidden_states, n=self.seq_len, dim=1, norm="forward")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=self.seq_len, dim=1, norm="forward")

        sequence_emb_fft = sequence_emb_fft[:, 0:seq_len, :]
        hidden_states = self.Dropout(sequence_emb_fft)
        hidden_states = hidden_states

        return hidden_states


class Attention(nn.Module):
    def __init__(self, emb_size, heads=2, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            emb_size, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.input_norm = nn.LayerNorm(emb_size)
        self.attention_norm = nn.LayerNorm(emb_size)
        self.mlp = FeedForward(emb_size, emb_size * 2, emb_size)

    def forward(self, q, kv=None, key_padding_mask=None) -> torch.Tensor:
        res = q
        h = self.input_norm(q)
        if kv is not None:
            h = self.attention(
                query=h, key=kv, value=kv, key_padding_mask=key_padding_mask
            )[0]
        else:
            h = self.attention(
                query=h, key=h, value=h, key_padding_mask=key_padding_mask
            )[0]
        h = res + h
        res = h
        h = self.attention_norm(h)
        h = self.mlp(h)
        return res + h


class MeanConv(MessagePassing):
    def __init__(
        self,
        emb_size,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        **kwargs,
    ):
        super().__init__(aggr, **kwargs)
        self.emb_size = emb_size
        self.w2 = nn.Linear(emb_size, emb_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = x.reshape(x.shape[0], -1)
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        n, _ = out.shape
        out = out.reshape(n, -1, self.emb_size)
        return self.w2(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr})"
        )


class TransformerConv(MessagePassing):
    r"""The graph transformer operator from the `"Masked Label Prediction:
    Unified Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper
    """
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.lin_out = Linear(heads * out_channels, heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        r"""Runs the forward pass of the module.

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )

        alpha = self._alpha
        self._alpha = None
        out = self.lin_out(out.view(-1, self.heads * self.out_channels))
        return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        # alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        # alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.tanh((query_i * key_j).sum(dim=-1))
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        return h


class NodeCoPredictor(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w_s = Linear(in_dim, in_dim)
        self.w_d = Linear(in_dim, in_dim)

    def forward(self, src, dst):
        o = torch.einsum("bd,ld->bl", self.w_s(src), self.w_d(dst))
        return o
