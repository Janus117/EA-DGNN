"""
implement persistant forecast as baseline for the node prop pred task
simply predict last seen label for the node
"""
from datetime import datetime
import timeit
import numpy as np
from torch_geometric.loader import TemporalDataLoader

# local imports
from DNAT.datasets.dataset_pyg import PyGNodePropPredDataset
from DNAT.datasets.utils.heuristics import MovingAverage
from sklearn.metrics import ndcg_score

from tqdm import tqdm

device = "cpu"

window = 7
name = "tgbn-genre"
dataset = PyGNodePropPredDataset(name=name, root="datasets")
num_classes = dataset.num_classes
data = dataset.get_TemporalData()
data = data.to(device)
forecaster = MovingAverage(data.num_nodes, num_classes, window=window)


train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15
)
global current_label_t
batch_size = 200

train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)
import torch


def test_n_update(loader):
    # label_t = dataset.get_label_time()  # check when does the first label start
    global current_label_t
    num_label_ts = 0
    total_score = 0
    tmp_src = torch.empty((0,)).to(device).int()
    tmp_dst = torch.empty((0,)).to(device).int()
    tmp_msg = torch.empty((0,)).to(device).float()
    for batch in tqdm(loader):
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        msg.squeeze_(-1)
        while src.shape[0] != 0:
            if t[-1] <= current_label_t:
                tmp_src = torch.cat((tmp_src, src))
                tmp_dst = torch.cat((tmp_dst, dst))
                tmp_msg = torch.cat((tmp_msg, msg))
                break
            else:
                tmp_src = torch.cat((tmp_src, src[t <= current_label_t]))
                tmp_dst = torch.cat((tmp_dst, dst[t <= current_label_t]))
                tmp_msg = torch.cat((tmp_msg, msg[t <= current_label_t]))
                forecaster.update_weight(tmp_src, tmp_dst, tmp_msg)
                src, dst, t, msg = (
                    src[t > current_label_t],
                    dst[t > current_label_t],
                    t[t > current_label_t],
                    msg[t > current_label_t],
                )
                tmp_src = torch.empty((0,)).to(device).int()
                tmp_dst = torch.empty((0,)).to(device).int()
                tmp_msg = torch.empty((0,)).to(device).float()

                label_tuple = dataset.get_node_label(current_label_t)
                current_label_t += day_dur
                if label_tuple is None:
                    continue
                label_ts, label_srcs, labels = (
                    label_tuple[0],
                    label_tuple[1],
                    label_tuple[2],
                )
                label_srcs = label_srcs.numpy()
                labels = labels.numpy()
                preds = []
                for i in range(0, label_srcs.shape[0]):
                    node_id = label_srcs[i]
                    pred_vec = forecaster.query_dict(node_id)
                    preds.append(pred_vec)

                np_pred = np.stack(preds, axis=0)
                np_true = labels
                score = ndcg_score(np_true, np_pred, k=10)
                total_score += score * label_srcs.shape[0]
                num_label_ts += label_srcs.shape[0]

    metric_dict = {
        "NDCG@10": total_score / num_label_ts,
    }
    return metric_dict


"""
train, val and test for one epoch only
"""
import calendar

day_dur = 60 * 60 * 24

current_label_t = list(dataset.dataset.label_dict.keys())[0]


start_time = timeit.default_timer()
metric_dict = test_n_update(train_loader)
print(metric_dict)
print(
    "Moving average on Training takes--- %s seconds ---"
    % (timeit.default_timer() - start_time)
)
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
dataset.reset_label_time()
# import pandas as pd
# n = pd.DataFrame(top10_total_score, columns=['srcs','score','ts'])
# n.to_csv('./tgbn-genre_score.csv', index=False)
