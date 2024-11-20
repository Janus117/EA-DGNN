import torch
from torch_geometric.utils import to_undirected
import numpy as np
import random

def log_string(log, string):
    log.write(str(string) + "\n")
    log.flush()
    print(string)


def set_random_seed(random_seed: int):
    r"""
    set random seed for reproducibility
    Args:
        random_seed (int): random seed
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'INFO: fixed random seed: {random_seed}')


def second_hop_edge_index(edge_index, edge_weight):
    num_nodes = edge_index.max().item() + 1
    edge_index, edge_weight = to_undirected(
        edge_index, edge_attr=edge_weight, num_nodes=num_nodes
    )
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

    second_hop_adj = torch.sparse.mm(adj, adj)
    second_hop_edge_index = second_hop_adj._indices()
    second_hop_edge_weight = second_hop_adj._values()

    # Remove self-loops
    mask = second_hop_edge_index[0] != second_hop_edge_index[1]
    second_hop_edge_index = second_hop_edge_index[:, mask]
    second_hop_edge_weight = second_hop_edge_weight[mask]

    return second_hop_edge_index, second_hop_edge_weight

def cat_ndcg_score(y_true, y_score, k):
    def _tie_averaged_dcg(y_true, y_score, discount_cumsum):
        _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
        ranked = np.zeros(len(counts))
        np.add.at(ranked, inv, y_true)
        ranked /= counts
        groups = np.cumsum(counts) - 1
        discount_sums = np.empty(len(counts))
        discount_sums[0] = discount_cumsum[groups[0]]
        discount_sums[1:] = np.diff(discount_cumsum[groups])
        return (ranked * discount_sums).sum()

    def _dcg_sample_scores(y_true, y_score, k=None, log_base=2, ignore_ties=False):
        discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
        if k is not None:
            discount[k:] = 0
        if ignore_ties:
            ranking = np.argsort(y_score)[:, ::-1]
            ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
            cumulative_gains = discount.dot(ranked.T)
        else:
            discount_cumsum = np.cumsum(discount)
            cumulative_gains = [
                _tie_averaged_dcg(y_t, y_s, discount_cumsum)
                for y_t, y_s in zip(y_true, y_score)
            ]
            cumulative_gains = np.asarray(cumulative_gains)
        return cumulative_gains

    gain = _dcg_sample_scores(y_true, y_score, k, ignore_ties=False)
    normalizing_gain = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return gain
