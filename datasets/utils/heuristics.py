import numpy as np
from torch_scatter import scatter
import torch


class MovingAverage:
    def __init__(self, nodes, num_class, window=7):
        self.dict = np.zeros((nodes, num_class))
        self.window = window

    @torch.no_grad()
    def update_weight(self, src, dst, msg):
        if src.shape[0] == 0:
            return
        uni_src, _ = torch.unique(src, sorted=True, return_counts=True)
        uni_dst, _ = torch.unique(dst, sorted=True, return_counts=True)
        mapping_src = {u.item(): idx for idx, u in enumerate(uni_src)}
        mapping_dst = {u.item(): idx for idx, u in enumerate(uni_dst)}
        src_idx = torch.tensor([mapping_src[s.item()] for s in src])
        dst_idx = torch.tensor([mapping_dst[d.item()] for d in dst])
        idx = src_idx * uni_dst.shape[0] + dst_idx
        dim_size = uni_dst.shape[0] * uni_src.shape[0]
        edge_batch_agg = scatter(msg, idx, dim_size=dim_size, reduce="sum")
        idx_nonzero = torch.nonzero(edge_batch_agg, as_tuple=True)[0]
        np_edge_batch_agg = edge_batch_agg.cpu().detach().numpy()
        np_uni_src = uni_src.cpu().detach().numpy()
        np_uni_dst = uni_dst.cpu().detach().numpy()
        np_idx_nonzero = idx_nonzero.cpu().detach().numpy()
        dst = np_uni_dst[np_idx_nonzero % np_uni_dst.shape[0]]
        src = np_uni_src[np_idx_nonzero // np_uni_dst.shape[0]]
        self.dict[src, dst] = (
            self.dict[src, dst] * (self.window - 1) / self.window
            + np_edge_batch_agg[np_idx_nonzero] / self.window
        )

    def query_dict(self, node_id):
        return self.dict[node_id]


class Persistent:
    def __init__(self, nodes, num_class):
        self.dict = np.zeros((nodes, num_class))

    @torch.no_grad()
    def update_weight(self, src, dst, msg):
        if src.shape[0] == 0:
            return
        uni_src, _ = torch.unique(src, sorted=True, return_counts=True)
        uni_dst, _ = torch.unique(dst, sorted=True, return_counts=True)
        mapping_src = {u.item(): idx for idx, u in enumerate(uni_src)}
        mapping_dst = {u.item(): idx for idx, u in enumerate(uni_dst)}
        src_idx = torch.tensor([mapping_src[s.item()] for s in src], device=src.device)
        dst_idx = torch.tensor([mapping_dst[d.item()] for d in dst], device=src.device)
        idx = src_idx * uni_dst.shape[0] + dst_idx
        dim_size = uni_dst.shape[0] * uni_src.shape[0]
        edge_batch_agg = scatter(msg, idx, dim_size=dim_size, reduce="sum")
        idx_nonzero = torch.nonzero(edge_batch_agg, as_tuple=True)[0]
        _, dst_sp_degrees = torch.unique(
            uni_dst[idx_nonzero % uni_dst.shape[0]], sorted=True, return_counts=True
        )
        np_edge_batch_agg = edge_batch_agg.cpu().detach().numpy()
        np_uni_src = uni_src.cpu().detach().numpy()
        np_uni_dst = uni_dst.cpu().detach().numpy()
        np_idx_nonzero = idx_nonzero.cpu().detach().numpy()
        dst = np_uni_dst[np_idx_nonzero % np_uni_dst.shape[0]]
        src = np_uni_src[np_idx_nonzero // np_uni_dst.shape[0]]
        self.dict[src, dst] = np_edge_batch_agg[np_idx_nonzero]

    def query_dict(self, node_id):
        return self.dict[node_id]


class Stats:
    def __init__(self, nodes, num_class):
        self.num_class = num_class
        self.nodes_history = torch.zeros((nodes - num_class, num_class, 28)).float()
        self.node_timestamp = torch.zeros((nodes - num_class, num_class)).float()

    @torch.no_grad()
    def update_weight(self, src, dst, msg, t):
        if src.shape[0] == 0:
            return
        uni_src, _ = torch.unique(src, sorted=True, return_counts=True)
        uni_dst, _ = torch.unique(dst, sorted=True, return_counts=True)
        mapping_src = {u.item(): idx for idx, u in enumerate(uni_src)}
        mapping_dst = {u.item(): idx for idx, u in enumerate(uni_dst)}
        src_idx = torch.tensor([mapping_src[s.item()] for s in src], device=src.device)
        dst_idx = torch.tensor([mapping_dst[d.item()] for d in dst], device=src.device)
        idx = src_idx * uni_dst.shape[0] + dst_idx
        dim_size = uni_dst.shape[0] * uni_src.shape[0]
        edge_batch_agg = scatter(msg, idx, dim_size=dim_size, reduce="sum")
        idx_nonzero = torch.nonzero(edge_batch_agg, as_tuple=True)[0]
        d = uni_dst[idx_nonzero % uni_dst.shape[0]]
        s = uni_src[idx_nonzero // uni_dst.shape[0]] - self.num_class
        self.nodes_history[s, d, 0:-1] = self.nodes_history[s, d, 1:]
        self.nodes_history[s, d, -1] = edge_batch_agg[idx_nonzero]
        self.node_timestamp[s, d] = t


class PopularityStats:
    def __init__(self, nodes, num_class):
        self.num_class = num_class
        self.dst_history = torch.zeros((num_class, 0))

    @torch.no_grad()
    def update_weight(self, src, dst, msg, t):
        if src.shape[0] == 0:
            return
        uni_dst, _ = torch.unique(dst, sorted=True, return_counts=True)
        mapping_dst = {u.item(): idx for idx, u in enumerate(uni_dst)}
        dst_idx = torch.tensor([mapping_dst[d.item()] for d in dst], device=src.device)
        idx = dst_idx
        dim_size = uni_dst.shape[0]
        dst_sum = scatter(msg, idx, dim_size=dim_size, reduce="mean")
        idx_nonzero = torch.nonzero(dst_sum, as_tuple=True)[0]
        dst_history = torch.zeros((self.num_class, 1))
        dst_history[uni_dst[idx_nonzero], 0] = dst_sum[idx_nonzero]
        self.dst_history = torch.cat((self.dst_history, dst_history), dim=-1)
