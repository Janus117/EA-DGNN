from .BaseModule import TimeEncode, FeedForward, Attention, FilterLayer

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class AttentionModel(torch.nn.Module):
    # 'TGAT', 'TCL', 'DyGformer', 'GraphMixer'
    def __init__(
        self,
        model_name,
        num_classes,
        time_feat_dim,
        hidden_dim,
        raw_msg_dim,
        neighbor_num,
        patch_size=2,
        num_layer=2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.model_name = model_name
        self.num_layers = num_layer
        self.neighbor_num = neighbor_num
        self.hidden_dim = hidden_dim
        self.edge_dim = time_feat_dim + hidden_dim
        self.raw_msg_dim = raw_msg_dim
        self.time_encoder = TimeEncode(emb_size=time_feat_dim)
        self.w_e = nn.Linear(raw_msg_dim, hidden_dim)
        self.ffn = nn.ModuleList(
            [
                FeedForward(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim * 2,
                    out_channels=hidden_dim,
                )
                for _ in range(num_layer)
            ]
        )
        if model_name == "TGAT":
            self.conv = nn.ModuleList(
                [
                    TransformerConv(
                        hidden_dim,
                        hidden_dim // 2,
                        heads=2,
                        dropout=0.1,
                        edge_dim=self.edge_dim,
                    )
                    for _ in range(num_layer)
                ]
            )
        if model_name in ["GraphMixer", "FreeDyg"]:
            self.ffn2 = nn.ModuleList(
                [
                    FeedForward(
                        in_channels=neighbor_num,
                        hidden_channels=hidden_dim,
                        out_channels=neighbor_num,
                    )
                    for _ in range(num_layer)
                ]
            )
        if model_name in ["TGAT", "GraphMixer", "FreeDyg"]:
            self.pre_norm = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layer)]
            )
            self.post_norm = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layer)]
            )

        if model_name == "TCL":
            self.dst_embedding = nn.Embedding(num_classes, hidden_dim)
            self.depth_embedding = nn.Embedding(neighbor_num, hidden_dim)
        if model_name in ["TCL", "DyGFormer"]:
            self.attention = nn.ModuleList(
                [
                    Attention(
                        emb_size=hidden_dim,
                    )
                    for _ in range(num_layer)
                ]
            )
        if model_name == "FreeDyg":
            self.filter = FilterLayer(seq_len=neighbor_num, hidden_dim=hidden_dim)

    def forward(self, x):
        pass

    def tgat_computing(self, n_id, mem_edge_index, node_t, msg, edge_t):
        node_t = self.time_encoder(
            torch.tensor(node_t, device=n_id.device).float()
        ).expand(n_id.size(0), -1)
        node_embedding = node_t
        edge_t = self.time_encoder(edge_t.float())
        edge_attr = torch.cat([edge_t, self.w_e(msg)], dim=-1)
        for i in range(self.num_layers):
            node_embedding = self.conv[i](
                self.pre_norm[i](node_embedding), mem_edge_index, edge_attr
            )
            node_embedding = node_embedding + self.ffn[i](
                self.post_norm[i](node_embedding)
            )
        return node_embedding

    def graphmixer_computing(self, batch_e_id, valid_edges_msg, valid_edges_t):
        n_id_size = batch_e_id.size(0)
        neighbors_num = batch_e_id.size(1)
        edges_msg = torch.zeros(
            (n_id_size, neighbors_num, self.raw_msg_dim),
            device=batch_e_id.device,
        ).float()
        edges_msg[batch_e_id > 0, :] = valid_edges_msg
        edges_msg = self.w_e(edges_msg)
        edges_t = torch.zeros(
            (n_id_size, neighbors_num, 1),
            device=batch_e_id.device,
        ).float()
        edges_t[batch_e_id > 0, :] = valid_edges_t.unsqueeze(-1).float()
        edges_t = self.time_encoder(edges_t).reshape(n_id_size, neighbors_num, -1)
        res = edges_msg + edges_t
        if self.model_name == "FreeDyg":
            res = res + self.filter(res)
        for i in range(self.num_layers):
            res = res + self.ffn2[i](self.pre_norm[i](res).permute(0, 2, 1)).permute(
                0, 2, 1
            )
            res = res + self.ffn[i](self.post_norm[i](res))
        res = torch.mean(res, dim=1)
        return res

    def dyg_computing(self, batch_e_id, valid_edges_msg, valid_edges_t):
        n_id_size = batch_e_id.size(0)
        neighbors_num = batch_e_id.size(1)
        patch_neighbors_num = neighbors_num // self.patch_size
        edges_msg = torch.zeros(
            (n_id_size, neighbors_num, self.raw_msg_dim),
            device=batch_e_id.device,
        ).float()
        edges_msg[batch_e_id > 0, :] = valid_edges_msg
        edges_msg = self.w_e(edges_msg)
        edges_t = torch.zeros(
            (n_id_size, neighbors_num, 1),
            device=batch_e_id.device,
        ).float()
        edges_t[batch_e_id > 0, :] = valid_edges_t.unsqueeze(-1).float()
        edges_t = self.time_encoder(edges_t).reshape(n_id_size, neighbors_num, -1)
        res = edges_msg + edges_t
        res = res.reshape(n_id_size, patch_neighbors_num, self.patch_size, -1)
        res = torch.mean(res, dim=-2)
        for i in range(self.num_layers):
            res = self.attention[i](res)
        res = torch.mean(res, dim=1)
        return res

    def tcl_computing(self, batch_n_id, batch_e_id, valid_edges_msg, valid_edges_t):
        mask = batch_e_id > 0
        n_id_size = batch_e_id.size(0)
        neighbors_num = batch_e_id.size(1)
        dst_nodes_embedding = torch.zeros(
            (n_id_size, neighbors_num, self.hidden_dim),
            device=batch_e_id.device,
        ).float()
        dst_nodes_embedding[mask, :] = self.dst_embedding(batch_n_id[mask])
        depth_embedding = torch.zeros(
            (n_id_size, neighbors_num, self.hidden_dim),
            device=batch_e_id.device,
        ).float()
        edge_size = (batch_e_id > 0).sum(dim=-1).cpu()
        ranges = torch.cat([torch.arange(n) for n in edge_size])
        depth_embedding[mask, :] = self.dst_embedding(ranges.to(batch_n_id.device))
        edges_msg = torch.zeros(
            (n_id_size, neighbors_num, self.raw_msg_dim),
            device=batch_e_id.device,
        ).float()
        edges_msg[batch_e_id > 0, :] = valid_edges_msg
        edges_msg = self.w_e(edges_msg)
        edges_t = torch.zeros(
            (n_id_size, neighbors_num, 1),
            device=batch_e_id.device,
        ).float()
        edges_t[batch_e_id > 0, :] = valid_edges_t.unsqueeze(-1).float()
        edges_t = self.time_encoder(edges_t).reshape(n_id_size, neighbors_num, -1)
        res = edges_msg + edges_t + dst_nodes_embedding + depth_embedding
        for i in range(self.num_layers):
            res = self.attention[i](res)
        res = torch.mean(res, dim=1)
        return res
