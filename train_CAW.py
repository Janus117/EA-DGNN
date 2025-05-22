from tqdm import tqdm
import torch
import torch.nn as nn
import timeit
import argparse
import matplotlib.pyplot as plt

from torch_geometric.loader import TemporalDataLoader
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.CWAN import CAWN
from model.BaseModule import NodePredictor
from datasets.dataset_pyg import PyGNodePropPredDataset
from sklearn.metrics import ndcg_score
from utils.utils import set_random_seed, log_string
from utils.neighbor_loader import AllNeighborSampler, Data, get_neighbor_sampler
import numpy as np

parser = argparse.ArgumentParser(
    description="parsing command line arguments as hyperparameters"
)
parser.add_argument("-s", "--seed", type=int, default=1, help="random seed to use")
parser.add_argument(
    "--model_name",
    type=str,
    default="CWAN",
    choices=["CWAN"],
)
parser.add_argument("--dataset", type=str, default="tgbn-trade", help="")

args = parser.parse_args()

model_name = args.model_name

duration = 1 if args.dataset == "tgbn-trade" else 60 * 60 * 24
# setting random seed
seed = int(args.seed)  # 1,2,3,4,5

torch.manual_seed(seed)
set_random_seed(seed)

# hyperparameters
lr = 0.0001
epochs = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = PyGNodePropPredDataset(name=args.dataset, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

num_classes = dataset.num_classes

data = dataset.get_TemporalData()
data = data.to(device)
now = datetime.now().strftime("%Y-%m-%d-%H-%M")
os.makedirs(f"./logs/{model_name}/{args.dataset}", exist_ok=True)
log = open(
    os.path.abspath(os.path.join(__file__, os.pardir))
    + "/logs/"
    + model_name
    + "/"
    + args.dataset
    + "/"
    + now
    + ".log",
    "w",
)
log_string(log, args)
log_string(log, f"setting random seed to be {seed}")
log_string(log, device)
eps = 1e-05
data.src = data.src + 1
data.dst = data.dst + 1
train_data = data[train_mask]
train_msg_mu = train_data.msg.mean(dim=0)
train_msg_std = train_data.msg.std(dim=0)
train_data.msg = (train_data.msg - train_msg_mu) / (train_msg_std + eps)
val_data = data[val_mask]
test_data = data[test_mask]
val_data.msg = (val_data.msg - train_msg_mu) / (train_msg_std + eps)
test_data.msg = (test_data.msg - train_msg_mu) / (train_msg_std + eps)
batch_size = 200
query_batch_size = 128
train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

neighbor_num = 20

np_data = dataset.dataset.full_data
src_node_ids = np_data["sources"].astype(np.int64)
dst_node_ids = np_data["destinations"].astype(np.int64)
node_interact_times = np_data["timestamps"].astype(np.float32)
edge_ids = np_data["edge_idxs"].astype(np.longlong)
labels = np_data["edge_label"]
edge_raw_features = np_data["edge_feat"].astype(np.float32)
edge_raw_features = np.vstack(
    [np.zeros(edge_raw_features.shape[1])[np.newaxis, :], edge_raw_features]
)
src_node_ids = src_node_ids + 1
dst_node_ids = dst_node_ids + 1
edge_ids = edge_ids + 1
full_data = Data(
    src_node_ids=src_node_ids,
    dst_node_ids=dst_node_ids,
    node_interact_times=node_interact_times,
    edge_ids=edge_ids,
    labels=labels,
)
neighbor_loader = get_neighbor_sampler(
    data=full_data,
    sample_neighbor_strategy="recent",
    seed=1,
)

memory_dim = time_dim = embedding_dim = 100
cawn = CAWN(
    edge_raw_features=edge_raw_features,
    neighbor_sampler=neighbor_loader,
    position_feat_dim=embedding_dim,
    time_feat_dim=time_dim,
    output_dim=embedding_dim,
    device=device,
    walk_length=2,
).to(device)

node_pred = NodePredictor(in_dim=embedding_dim, out_dim=num_classes).to(device)
optimizer = torch.optim.AdamW(
    set(cawn.parameters()) | set(node_pred.parameters()),
    lr=lr,
)

criterion = torch.nn.CrossEntropyLoss()


current_label_t = list(dataset.dataset.label_dict.keys())[0]


def train():
    global current_label_t
    cawn.train()
    node_pred.train()  # Start with an empty graph.
    total_loss = 0
    total_score = 0
    num_label = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)

        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        while src.shape[0] != 0:
            if t[-1] < current_label_t:
                break
            else:
                label_tuple = dataset.get_node_label(current_label_t)
                if label_tuple is None:
                    current_label_t += duration
                    continue
                label_ts, label_srcs, labels = (
                    label_tuple[0],
                    label_tuple[1],
                    label_tuple[2],
                )
                label_srcs = label_srcs + 1
                label_srcs = label_srcs.to(device)
                labels = labels.to(device)
                if label_srcs.size(0) > query_batch_size:
                    preds = label_srcs.new_zeros((0, num_classes))
                    step = label_srcs.size(0) // query_batch_size
                    if label_srcs.size(0) % query_batch_size > 0:
                        step += 1
                    for i in range(step):
                        ls = label_srcs[
                            i * query_batch_size : (i + 1) * query_batch_size
                        ]
                        l = labels[i * query_batch_size : (i + 1) * query_batch_size]

                        optimizer.zero_grad()
                        n_id = ls
                        n_id = n_id.detach().cpu().numpy()
                        node_interact_times = np.empty_like(n_id)
                        node_interact_times[...] = current_label_t
                        node_interact_times = node_interact_times.astype(np.float32)
                        z = cawn.compute_src_dst_node_temporal_embeddings(
                            src_node_ids=n_id,
                            node_interact_times=node_interact_times,
                            num_neighbors=neighbor_num,
                        )
                        pred = node_pred(z)
                        optimizer.zero_grad()
                        loss = criterion(pred, l)
                        loss.backward()
                        optimizer.step()
                        preds = torch.cat((preds, pred), dim=0)
                        total_loss += float(loss) * ls.shape[0]

                else:
                    n_id = label_srcs
                    n_id = n_id.detach().cpu().numpy()
                    node_interact_times = np.empty_like(n_id)
                    node_interact_times[...] = current_label_t
                    node_interact_times = node_interact_times.astype(np.float32)
                    z = cawn.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=n_id,
                        node_interact_times=node_interact_times,
                        num_neighbors=neighbor_num,
                    )
                    optimizer.zero_grad()
                    preds = node_pred(z)
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss) * label_srcs.shape[0]
                np_pred = preds.cpu().detach().numpy()
                np_true = labels.cpu().detach().numpy()
                current_label_t += duration
                score = ndcg_score(np_true, np_pred, k=10)
                total_score += score * label_srcs.shape[0]
                num_label += label_srcs.shape[0]

    metric_dict = {
        "ce": total_loss / num_label,
        "NDCG@10": total_score / num_label,
    }
    return metric_dict


@torch.no_grad()
def test(loader):
    global current_label_t
    cawn.eval()
    node_pred.eval()
    total_score = 0
    num_label = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        while src.shape[0] != 0:
            if t[-1] < current_label_t:
                break
            else:
                label_tuple = dataset.get_node_label(current_label_t)

                if label_tuple is None:
                    current_label_t += duration
                    continue
                label_ts, label_srcs, labels = (
                    label_tuple[0],
                    label_tuple[1],
                    label_tuple[2],
                )
                label_srcs = label_srcs + 1
                label_srcs = label_srcs.to(device)
                labels = labels.to(device)
                if label_srcs.size(0) > query_batch_size:
                    preds = label_srcs.new_zeros((0, num_classes))
                    step = label_srcs.size(0) // query_batch_size
                    if label_srcs.size(0) % query_batch_size > 0:
                        step += 1
                    for i in range(step):
                        ls = label_srcs[
                            i * query_batch_size : (i + 1) * query_batch_size
                        ]

                        n_id = ls
                        n_id = n_id.detach().cpu().numpy()
                        node_interact_times = np.empty_like(n_id)
                        node_interact_times[...] = current_label_t
                        node_interact_times = node_interact_times.astype(np.float32)
                        z = cawn.compute_src_dst_node_temporal_embeddings(
                            src_node_ids=n_id,
                            node_interact_times=node_interact_times,
                            num_neighbors=neighbor_num,
                        )
                        pred = node_pred(z)
                        preds = torch.cat((preds, pred), dim=0)

                else:
                    n_id = label_srcs
                    n_id = n_id.detach().cpu().numpy()
                    node_interact_times = np.empty_like(n_id)
                    node_interact_times[...] = current_label_t
                    node_interact_times = node_interact_times.astype(np.float32)
                    z = cawn.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=n_id,
                        node_interact_times=node_interact_times,
                        num_neighbors=neighbor_num,
                    )
                    preds = node_pred(z)
                np_pred = preds.cpu().detach().numpy()
                np_true = labels.cpu().detach().numpy()
                current_label_t += duration
                score = ndcg_score(np_true, np_pred, k=10)
                total_score += score * label_srcs.shape[0]
                num_label += label_srcs.shape[0]

    metric_dict = {
        "NDCG@10": total_score / num_label,
    }
    return metric_dict


import itertools

num_params = sum(
    p.numel()
    for p in itertools.chain(cawn.parameters(), node_pred.parameters())
    if p.requires_grad
)
log_string(log, f"Number of parameters in the model: {num_params}")


max_test_score = 0  # find the best test score based on validation score
max_val_score = 0
best_test_idx = 0
for epoch in range(1, epochs + 1):
    current_label_t = list(dataset.dataset.label_dict.keys())[0]
    start_time = timeit.default_timer()
    train_dict = train()
    log_string(log, "------------------------------------")
    log_string(log, f"training Epoch: {epoch:02d}")
    log_string(log, train_dict)
    log_string(
        log, "Training takes--- %s seconds ---" % (timeit.default_timer() - start_time)
    )

    start_time = timeit.default_timer()
    val_dict = test(val_loader)
    log_string(log, val_dict)
    log_string(
        log,
        "Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time),
    )
    start_time = timeit.default_timer()
    test_dict = test(test_loader)
    log_string(log, test_dict)
    if test_dict["NDCG@10"] > max_test_score:
        max_test_score = test_dict["NDCG@10"]
        max_val_score = val_dict["NDCG@10"]
        best_test_idx = epoch - 1
    log_string(
        log, "Test takes--- %s seconds ---" % (timeit.default_timer() - start_time)
    )
    log_string(log, "------------------------------------")
    dataset.reset_label_time()

log_string(log, "------------------------------------")
log_string(log, "------------------------------------")
log_string(log, f"best test epoch   : {best_test_idx + 1}")
log_string(log, f"best val score: {max_val_score}")
log_string(log, f"best test score: {max_test_score}")
