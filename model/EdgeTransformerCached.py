import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_scatter import scatter

from .BaseModule import TimeEncode, Attention, FeedForward, MeanConv, TransformerConv
import math
from DNAT.utils.utils import second_hop_edge_index
from torch_geometric.nn.inits import zeros
from DNAT.utils.neighbor_loader import LastNeighborLoader
import timeit


class EdgeTransformer(torch.nn.Module):
    def __init__(
        self,
        device,
        num_classes,
        num_nodes,
        datasets_name,
        emb_size=16,
        history_length=28,
        msg_threshold=0.1,
        dropout=0.2,
        user_neighbor_num=10,
        item_neighbor_num=10,
        second_src_degrees_threshold=1.0,
        second_dst_degrees_threshold=1.0,
        K=10,
        time_temperature=1,
        wo_time_embed=False,
        wo_aux_embed=False,
        wo_src_spatial=False,
        wo_dst_spatial=False,
        wo_aggregation=False,
        enable_linear=True,
        num_layers=2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.enable_linear = enable_linear
        self.wo_time_embed = wo_time_embed
        self.wo_aux_embed = wo_aux_embed
        self.wo_src_spatial = wo_src_spatial
        self.wo_dst_spatial = wo_dst_spatial
        self.wo_aggregation = wo_aggregation
        self.offset = 0 if datasets_name == "tgbn-trade" else num_classes
        self.act = nn.SiLU()
        self.K = K
        self.time_temperature = time_temperature
        self.second_dst_degrees_threshold = second_dst_degrees_threshold
        self.second_src_degrees_threshold = second_src_degrees_threshold
        self.user_neighbor_num = user_neighbor_num
        self.item_neighbor_num = item_neighbor_num
        self.msg_threshold = msg_threshold
        self.msg_ts_threshold = (
            history_length
            if datasets_name == "tgbn-trade"
            else 60 * 60 * 24 * history_length
        )
        self.datasets_name = datasets_name
        self.device = device
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.emb_size = emb_size

        self.msg_emb_size = emb_size
        self.history_length = history_length
        self.msg_len = 3 if self.datasets_name == "tgbn-reddit" else 2
        if self.wo_aux_embed:
            self.msg_len -= 1
        self.dst_state_size = emb_size
        if self.wo_dst_spatial:
            self.final_state_size = emb_size
        else:
            self.final_state_size = emb_size * 2
        self.attention_emb_size = emb_size
        if not self.wo_time_embed:
            self.dst_state_size += emb_size
            self.final_state_size += emb_size
            self.attention_emb_size += emb_size
            self.time_encoder = TimeEncode(emb_size)

        self.dropout = nn.Dropout(dropout)
        self.src_attention = nn.ModuleList(
            [
                Attention(
                    self.attention_emb_size,
                    heads=2,
                    dropout=dropout,
                    enable_linear=self.enable_linear,
                )
                for _ in range(num_layers)
            ]
        )
        self.dst_attention = nn.ModuleList(
            [
                Attention(
                    self.attention_emb_size,
                    heads=2,
                    dropout=dropout,
                    enable_linear=self.enable_linear,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_attention = nn.ModuleList(
            [
                Attention(
                    self.final_state_size,
                    heads=4,
                    dropout=dropout,
                    enable_linear=self.enable_linear,
                )
                for _ in range(num_layers)
            ]
        )
        self.src_conv = nn.ModuleList(
            [MeanConv(self.attention_emb_size) for _ in range(1)]
        )

        self.dst_conv = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=self.dst_state_size,
                    out_channels=self.dst_state_size // 4,
                    heads=4,
                    dropout=dropout,
                )
                for _ in range(1)
            ]
        )
        self.K_embedding = nn.Embedding(self.K, emb_size)
        self.empty_embedding = nn.Parameter(torch.empty(self.attention_emb_size))
        self.w_m = nn.Linear(self.msg_len, self.msg_emb_size)
        self.w_dm = nn.Linear(self.msg_len, self.msg_emb_size)

        self.w_q = nn.Linear(self.attention_emb_size, self.K)

        self.s_mlp = FeedForward(
            self.attention_emb_size,
            self.attention_emb_size * 2,
            self.attention_emb_size,
        )
        self.s_norm = nn.LayerNorm(self.attention_emb_size)
        self.s_mlp2 = FeedForward(
            self.attention_emb_size,
            self.attention_emb_size * 2,
            self.attention_emb_size,
        )
        self.s_norm2 = nn.LayerNorm(self.attention_emb_size)
        self.d_mlp = FeedForward(
            self.attention_emb_size,
            self.attention_emb_size * 2,
            self.attention_emb_size,
        )
        self.d_norm = nn.LayerNorm(self.attention_emb_size)

        self.d_mlp2 = FeedForward(
            self.dst_state_size, self.dst_state_size * 2, self.dst_state_size
        )
        self.d_norm2 = nn.LayerNorm(self.dst_state_size)
        # if self.fusion:
        #     self.d_mlp3 = FeedForward(
        #         self.dst_state_size, self.attention_emb_size, self.attention_emb_size
        #     )
        # else:
        self.d_mlp3 = FeedForward(
            self.dst_state_size, self.attention_emb_size, self.emb_size
        )
        self.d_norm3 = nn.LayerNorm(self.dst_state_size)

        self.final_mlp = FeedForward(
            self.final_state_size, self.final_state_size * 2, self.final_state_size
        )
        self.layer_norm = nn.LayerNorm(self.final_state_size)
        self.pred = nn.Linear(self.final_state_size, 1)
        self.reset_parameters()
        self.src_history = None
        self.src_history_timestamp = None
        self.src_history_mask = None

        self.dst_history = None
        self.dst_history_timestamp = None
        self.dst_history_mask = None

        self.user_neighbor_loader = LastNeighborLoader(
            self.num_nodes, size=user_neighbor_num, device=device
        )
        self.item_neighbor_loader = LastNeighborLoader(
            self.num_classes, size=item_neighbor_num, device=device
        )
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long, device=device)

    def reset_parameters(self):
        nn.init.normal_(self.empty_embedding)
        if not self.wo_time_embed:
            self.time_encoder.reset_parameters()

    def reset_state(self):
        if self.datasets_name == "tgbn-token":
            self.src_history = (
                torch.zeros(
                    (
                        self.num_nodes - self.offset,
                        self.num_classes,
                        self.history_length,
                        self.msg_len,
                    )
                )
                .float()
                .cpu()
            )
            self.src_history_timestamp = (
                torch.zeros(
                    (
                        self.num_nodes - self.offset,
                        self.num_classes,
                        self.history_length,
                    )
                )
                .int()
                .cpu()
            )
        else:
            self.src_history = (
                torch.zeros(
                    (
                        self.num_nodes - self.offset,
                        self.num_classes,
                        self.history_length,
                        self.msg_len,
                    )
                )
                .float()
                .to(self.device)
            )
            self.src_history_timestamp = (
                torch.zeros(
                    (
                        self.num_nodes - self.offset,
                        self.num_classes,
                        self.history_length,
                    )
                )
                .int()
                .to(self.device)
            )

        self.dst_history = (
            torch.zeros((self.num_classes, self.history_length, self.msg_len))
            .float()
            .to(self.device)
        )
        self.dst_history_timestamp = (
            torch.zeros((self.num_classes, self.history_length)).int().to(self.device)
        )

        self.user_neighbor_loader.reset_state()
        self.item_neighbor_loader.reset_state()

    @torch.no_grad()
    def save_batch(self, src, dst, ts, msg, aux_msg=None, data_dict=None):
        if data_dict is None:
            data_dict = {}
        if src.shape[0] == 0:
            return
        data_dict[ts] = {}
        uni_src, src_tem_degrees = torch.unique(src, sorted=True, return_counts=True)
        uni_dst, dst_tem_degrees = torch.unique(dst, sorted=True, return_counts=True)
        mapping_src = {u.item(): idx for idx, u in enumerate(uni_src)}
        mapping_dst = {u.item(): idx for idx, u in enumerate(uni_dst)}
        src_idx = torch.tensor([mapping_src[s.item()] for s in src], device=self.device)
        dst_idx = torch.tensor([mapping_dst[d.item()] for d in dst], device=self.device)
        idx = src_idx * uni_dst.shape[0] + dst_idx
        dim_size = uni_dst.shape[0] * uni_src.shape[0]
        uni_idx, edges_num = torch.unique(idx, sorted=True, return_counts=True)
        edge_batch_agg = scatter(msg, idx, dim_size=dim_size, reduce="max")
        # edge_batch_agg1 = scatter(msg, idx, dim_size=dim_size, reduce="sum")
        # edge_batch_agg2 = scatter(msg, idx, dim_size=dim_size, reduce="mean")
        idx_nonzero = torch.nonzero(edge_batch_agg, as_tuple=True)[0]
        idx_nonzero1 = torch.nonzero(
            edge_batch_agg > self.msg_threshold, as_tuple=True
        )[0]
        _, dst_sp_degrees = torch.unique(
            uni_dst[idx_nonzero % uni_dst.shape[0]], sorted=True, return_counts=True
        )
        _, src_sp_degrees = torch.unique(
            uni_src[idx_nonzero // uni_dst.shape[0]], sorted=True, return_counts=True
        )
        # print(dst_sp_degrees.max().item())
        dst_sp_degrees_norm = (dst_sp_degrees - dst_sp_degrees.min() + 1e-05) / (
            dst_sp_degrees.max() - dst_sp_degrees.min() + 1e-05
        )
        data_dict[ts]["idx_nonzero1"] = idx_nonzero1
        data_dict[ts]["src_sp_degrees"] = src_sp_degrees
        dst_agg = scatter(
            edge_batch_agg[idx_nonzero],
            uni_dst[idx_nonzero % uni_dst.shape[0]],
            reduce="mean",
        )
        dst_idx_nonzero = torch.nonzero(dst_agg, as_tuple=True)[0]
        dst_msg = torch.stack(
            (
                dst_agg[dst_idx_nonzero],
                dst_sp_degrees_norm,
            ),
            dim=-1,
        )

        if self.datasets_name == "tgbn-reddit":
            aux_edge_batch_agg = scatter(aux_msg, idx, dim_size=dim_size, reduce="mean")
            aux_dst_agg = scatter(
                aux_edge_batch_agg[idx_nonzero],
                uni_dst[idx_nonzero % uni_dst.shape[0]],
                reduce="mean",
            )
            dst_msg = torch.cat((dst_msg, aux_dst_agg[uni_dst].unsqueeze(-1)), dim=-1)

        data_dict[ts]["uni_dst"] = uni_dst
        data_dict[ts]["uni_src"] = uni_src
        data_dict[ts]["dst_msg"] = dst_msg

        d = uni_dst[idx_nonzero % uni_dst.shape[0]]
        s = uni_src[idx_nonzero // uni_dst.shape[0]] - self.offset
        src_msg = torch.stack(
            (
                edge_batch_agg[idx_nonzero],
                edges_num / src_tem_degrees[idx_nonzero // uni_dst.shape[0]],
            ),
            dim=-1,
        )
        if self.datasets_name == "tgbn-reddit":
            src_msg = torch.cat(
                (src_msg, aux_edge_batch_agg[idx_nonzero].unsqueeze(-1)), dim=-1
            )
        data_dict[ts]["s"] = s
        data_dict[ts]["d"] = d
        data_dict[ts]["src_msg"] = src_msg

    def update_batch(self, ts, data_dict):
        idx_nonzero1 = data_dict[ts]["idx_nonzero1"]
        uni_dst = data_dict[ts]["uni_dst"]
        uni_src = data_dict[ts]["uni_src"]
        dst_msg = data_dict[ts]["dst_msg"]
        src_msg = data_dict[ts]["src_msg"]
        s = data_dict[ts]["s"]
        d = data_dict[ts]["d"]
        src_sp_degrees= data_dict[ts]["src_sp_degrees"]
        if idx_nonzero1.shape[0] > 0:
            edge_index = torch.stack(
                (
                    uni_src[idx_nonzero1 // uni_dst.shape[0]],
                    uni_dst[idx_nonzero1 % uni_dst.shape[0]],
                ),
                dim=0,
            )
            second_hop, second_weight = second_hop_edge_index(
                edge_index, edge_index.new_ones(edge_index.size(1)).float()
            )
            if second_hop.shape[1] > 0:
                src_second_hop = second_hop[:, second_hop[0] >= self.num_classes]
                src_second_weight = second_weight[second_hop[0] >= self.num_classes]
                src_second_hop = src_second_hop[
                    :, src_second_weight > self.second_src_degrees_threshold
                ]
                src_second_weight = src_second_weight[
                    src_second_weight > self.second_src_degrees_threshold
                ]
                if src_second_hop.shape[1] > 0:
                    self.user_neighbor_loader.insert2(
                        src_second_hop[0],
                        src_second_hop[1],
                        uni_src,
                        src_sp_degrees,
                        src_second_weight,
                    )
                    # self.neighbor_loader.insert(src_second_hop[0], src_second_hop[1])

                dst_second_hop = second_hop[:, second_hop[0] < self.num_classes]
                dst_second_weight = second_weight[second_hop[0] < self.num_classes]
                dst_second_hop = dst_second_hop[
                    :, dst_second_weight > self.second_dst_degrees_threshold
                ]
                dst_second_weight = dst_second_weight[
                    dst_second_weight > self.second_dst_degrees_threshold
                ]
                if dst_second_hop.shape[1] > 0:
                    self.item_neighbor_loader.insert(
                        dst_second_hop[0], dst_second_hop[1], dst_second_weight
                    )
        self.dst_history[uni_dst] = torch.roll(
            self.dst_history[uni_dst], shifts=1, dims=-2
        )
        self.dst_history[uni_dst, 0, :] = dst_msg
        self.dst_history_timestamp[uni_dst] = torch.roll(
            self.dst_history_timestamp[uni_dst], shifts=1, dims=-1
        )
        self.dst_history_timestamp[uni_dst, 0] = ts

        if self.datasets_name == "tgbn-token":
            src_msg = src_msg.cpu()
            s = s.cpu()
            d = d.cpu()
        self.src_history[s, d] = torch.roll(
            self.src_history[s, d], shifts=1, dims=-2
        )
        self.src_history[s, d, 0, :] = src_msg
        self.src_history_timestamp[s, d] = torch.roll(
            self.src_history_timestamp[s, d], shifts=1, dims=-1
        )
        self.src_history_timestamp[s, d, 0] = ts

    def reg_loss(self, alpha, logit):
        log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(
            1, logit.size(1)
        )
        return -torch.mean(torch.sum(torch.mul(alpha, log_pi), dim=1))

    def dst_rep2(self, label_t):
        if self.wo_dst_spatial:
            return None, 0
        valid_dst = (
            label_t - self.dst_history_timestamp[:, 0]
        ) <= self.msg_ts_threshold
        valid_dst_num = valid_dst.sum().item()
        # valid_dst = (self.dst_history_timestamp[:, -1]) > 0
        dst_state = (
            torch.empty((self.num_classes, self.attention_emb_size))
            .float()
            .to(self.device)
        )
        time = self.dst_history_timestamp[valid_dst]

        time[time > 0] = label_t - time[time > 0]
        time = time * self.time_temperature
        if self.wo_time_embed:
            valid_dst_feature = self.w_dm(self.dst_history[valid_dst])
        else:
            valid_dst_feature = torch.cat(
                (
                    self.w_dm(self.dst_history[valid_dst]),
                    self.time_encoder(time.float()).reshape(
                        -1, self.history_length, self.emb_size
                    ),
                ),
                dim=-1,
            )
        valid_dst_feature = valid_dst_feature + self.d_mlp(
            self.d_norm(valid_dst_feature)
        )
        valid_dst_feature = self.dropout(valid_dst_feature)

        for i in range(self.num_layers):
            valid_dst_feature = self.dst_attention[i](
                q=valid_dst_feature,
                # key_padding_mask=self.dst_history_mask[valid_dst],
            )
        valid_dst_feature = valid_dst_feature.mean(dim=-2)
        dst_state[valid_dst] = valid_dst_feature
        dst_state[~valid_dst] = self.empty_embedding.unsqueeze(0).expand(
            (~valid_dst).sum().item(), self.attention_emb_size
        )
        loss = 0
        #
        # k_emb = self.K_embedding(torch.arange(self.K).to(self.device))
        # logit = self.w_q(dst_state)
        # if self.training:
        #     alpha = F.gumbel_softmax(
        #         logit,
        #         tau=1,
        #         dim=-1,
        #     )
        #     loss = self.reg_loss(alpha, logit)
        # else:
        #     alpha = F.softmax(
        #         logit,
        #         dim=-1,
        #     )
        # alpha = self.dropout(alpha)
        # dst_id = torch.matmul(alpha, k_emb)
        #
        # dst_state = torch.cat((dst_state, dst_id), dim=-1)

        dst_state = dst_state + self.d_mlp2(self.d_norm2(dst_state))
        dst_state = self.dropout(dst_state)

        n_id = torch.arange(self.num_classes).to(self.device)
        _, mem_edge_index = self.item_neighbor_loader(n_id)
        dst_state_hop1 = self.dst_conv[0](dst_state, mem_edge_index)
        dst_state = dst_state + dst_state_hop1

        dst_state = self.d_mlp3(self.d_norm3(dst_state))
        return dst_state, loss

    def src_rep2(self, label_srcs, label_t):
        n_id, mem_edge_index = self.user_neighbor_loader(label_srcs)
        # n_id = label_srcs
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        if self.datasets_name == "tgbn-token":
            n_id = n_id.cpu()
        valid_src = (
            (label_t - self.src_history_timestamp[n_id - self.offset, :, 0])
        ) <= self.msg_ts_threshold

        src_rep = (
            torch.empty((n_id.shape[0], self.num_classes, self.attention_emb_size))
            .float()
            .to(self.device)
        )
        time = self.src_history_timestamp[n_id - self.offset][valid_src].to(self.device)
        time[time > 0] = label_t - time[time > 0]
        time = time * self.time_temperature
        if self.wo_time_embed:
            valid_src_feature = self.w_m(
                self.src_history[n_id - self.offset][valid_src].to(self.device)
            )
        else:
            valid_src_feature = torch.cat(
                (
                    self.w_m(
                        self.src_history[n_id - self.offset][valid_src].to(self.device)
                    ),
                    self.time_encoder(time.float()).reshape(
                        -1, self.history_length, self.emb_size
                    ),
                ),
                dim=-1,
            )
        valid_src_feature = valid_src_feature + self.s_mlp(
            self.s_norm(valid_src_feature)
        )
        valid_src_feature = self.dropout(valid_src_feature)
        for i in range(self.num_layers):
            valid_src_feature = self.src_attention[i](
                q=valid_src_feature,
                # key_padding_mask=self.src_history_mask[n_id - self.offset][
                #     valid_src
                # ].to(self.device),
            )
        valid_src_feature = valid_src_feature.mean(dim=-2)
        src_rep[valid_src] = valid_src_feature
        src_rep[~valid_src] = self.empty_embedding.unsqueeze(0).expand(
            (~valid_src).sum().item(), src_rep.shape[-1]
        )
        # src_rep[~valid_src] = src_rep.new_zeros(
        #     ((~valid_src).sum().item(), src_rep.shape[-1])
        # )
        # if self.fusion:
        #     src_rep[~valid_src] = src_rep.new_zeros(
        #         ((~valid_src).sum().item(), src_rep.shape[-1])
        #     )
        #     src_rep = torch.sum(src_rep, dim=-2)
        #     src_rep = src_rep[self.assoc[label_srcs]]
        #     return src_rep
        if not self.wo_src_spatial:
            src_rep = src_rep + self.src_conv[0](src_rep, mem_edge_index)
        src_rep = src_rep[self.assoc[label_srcs]]
        if self.datasets_name == "tgbn-token":
            label_srcs = label_srcs.cpu()
        valid_src2 = (
            ((label_t - self.src_history_timestamp[label_srcs - self.offset, :, 0]))
            <= self.msg_ts_threshold
        ).to(self.device)

        src_rep[~valid_src2] = src_rep[~valid_src2] + self.empty_embedding.unsqueeze(
            0
        ).expand((~valid_src2).sum().item(), src_rep.shape[-1])
        src_rep = src_rep + self.s_mlp2(self.s_norm2(src_rep))
        src_rep = self.dropout(src_rep)

        return src_rep

    def forward(self, label_srcs, src_rep, dst_rep):
        # if self.fusion:
        #     return torch.matmul(src_rep, dst_rep.T)
        if self.wo_dst_spatial:
            final_rep = src_rep
        else:
            final_rep = torch.cat(
                (src_rep, dst_rep.unsqueeze(0).expand(label_srcs.shape[0], -1, -1)),
                dim=-1,
            )
        final_rep = final_rep + self.final_mlp(self.layer_norm(final_rep))
        final_rep = self.dropout(final_rep)
        for i in range(self.num_layers):
            final_rep = self.final_attention[i](q=final_rep)

        return self.pred(final_rep).squeeze(-1)
