"""
implement persistant forecast as baseline for the node prop pred task
simply predict last seen label for the node
"""
from datetime import datetime
import timeit
import numpy as np
from torch_geometric.loader import TemporalDataLoader
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
from datasets.dataset_pyg import PyGNodePropPredDataset
from heuristics import Persistent
from sklearn.metrics import ndcg_score
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
window = 7
name = "tgbn-token"
dataset = PyGNodePropPredDataset(name=name, root="datasets")
num_classes = dataset.num_classes
data = dataset.get_TemporalData()
data = data.to(device)
forecaster = Persistent(data.num_nodes, num_classes)


val_ratio = 0.15
test_ratio = 0.15

val_time, test_time = np.quantile(
    data.t.cpu().numpy(), [1.0 - val_ratio - test_ratio, 1.0 - test_ratio]
)
label_t_list = np.array(list(dataset.dataset.label_dict.keys()))
val_idx = int((data.t <= label_t_list[np.abs(label_t_list - val_time).argmin()]).sum())
test_idx = int((data.t <= label_t_list[np.abs(label_t_list - test_time).argmin()]).sum())

train_data, val_data, test_data = (
    data[:val_idx],
    data[val_idx:test_idx],
    data[test_idx:],
)

global current_label_t
batch_size = 200
# print(train_data.msg[:, 0].max())
# print(train_data.msg[:, 0].min())
# print(train_data.msg[:, 0].mean())
# print(train_data.msg[:, 0].median())

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)
import torch


def test_n_update(loader):
    # label_t = dataset.get_label_time()  # check when does the first label start
    global current_label_t
    num_label_ts = 0
    total_score = 0
    for batch in tqdm(loader):
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        msg = msg[:, 0]
        while src.shape[0] != 0:
            mask = t < current_label_t
            forecaster.update_weight(src[mask], dst[mask], msg[mask])
            src, dst, t, msg = (
                src[~mask],
                dst[~mask],
                t[~mask],
                msg[~mask],
            )
            if t.shape[0] == 0 or t[-1] < current_label_t:
                break
            else:
                label_tuple = dataset.get_node_label(current_label_t)

                current_label_t += day_dur
                if label_tuple is None:
                    continue
                label_ts, label_srcs, labels = (
                    label_tuple[0],
                    label_tuple[1],
                    label_tuple[2],
                )
                # print(label_ts[-1])
                label_srcs = label_srcs.numpy()
                labels = labels.numpy()
                np_pred = forecaster.query_dict(label_srcs)
                forecaster.dict[label_srcs] = 0
                np_true = labels
                score = ndcg_score(np_true, np_pred, k=10)
                # print(datetime.fromtimestamp(current_label_t),score)
                total_score += score * label_srcs.shape[0]
                num_label_ts += label_srcs.shape[0]
        if src.shape[0] != 0:
            forecaster.update_weight(src, dst, msg)
    metric_dict = {
        "NDCG@10": total_score / num_label_ts,
    }
    return metric_dict



"""
train, val and test for one epoch only
"""

day_dur = 60 * 60 * 24

current_label_t = list(dataset.dataset.label_dict.keys())[0]


start_time = timeit.default_timer()
metric_dict = test_n_update(train_loader)
print(metric_dict)
print(
    "Moving average on Training takes--- %s seconds ---"
    % (timeit.default_timer() - start_time)
)
# print(forecaster.edges.mean(), forecaster.edges.std())
# print(forecaster.dst.mean(), forecaster.dst.std())
start_time = timeit.default_timer()
val_dict = test_n_update(val_loader)
print(val_dict)
print(
    "Moving average on Validation takes--- %s seconds ---"
    % (timeit.default_timer() - start_time)
)

start_time = timeit.default_timer()
test_dict = test_n_update(test_loader)
print(test_dict)
print(
    "Moving average on Test takes--- %s seconds ---"
    % (timeit.default_timer() - start_time)
)
# dataset.reset_label_time()
# import pandas as pd
# n = pd.DataFrame(top10_total_score, columns=['srcs','score','ts'])
# n.to_csv('./tgbn-genre_score.csv', index=False)
