import timeit
import torch


import itertools
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
from sklearn.metrics import ndcg_score
import os
import argparse
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import log_string, cat_ndcg_score, set_random_seed
from datasets.dataset_pyg import PyGNodePropPredDataset

# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.EdgeTransformerCached import EdgeTransformer
from datetime import datetime

global current_label_t


def train(dataset, model, data_dict):
    global current_label_t
    model.train()
    model.reset_state()
    total_loss = 0
    num_label_ts = 0
    total_score = 0
    data_dict.keys()
    for _ in tqdm(list(data_dict.keys())):
        model.update_batch(current_label_t - duration, data_dict)
        label_tuple = dataset.get_node_label(current_label_t)
        if label_tuple is None:
            current_label_t += duration
            continue
        label_ts, label_srcs, labels = (
            label_tuple[0],
            label_tuple[1],
            label_tuple[2],
        )

        label_srcs = label_srcs.to(device)
        labels = labels.to(device)
        if label_srcs.size(0) > query_batch_size:
            preds = label_srcs.new_zeros((0, num_classes))
            step = label_srcs.size(0) // query_batch_size
            if label_srcs.size(0) % query_batch_size > 0:
                step += 1
            for i in range(step):
                dst_rep, reg_loss = model.dst_rep2(current_label_t)
                ls = label_srcs[i * query_batch_size : (i + 1) * query_batch_size]
                l = labels[i * query_batch_size : (i + 1) * query_batch_size]

                optimizer.zero_grad()

                src_rep = model.src_rep2(ls, current_label_t)
                pred = model(ls, src_rep, dst_rep)
                _loss = criterion(pred, l)
                _loss += reg_loss
                _loss.backward()
                optimizer.step()
                preds = torch.cat((preds, pred), dim=0)
                total_loss += float(_loss) * ls.shape[0]
        else:
            dst_rep, reg_loss = model.dst_rep2(current_label_t)
            src_rep = model.src_rep2(label_srcs, current_label_t)
            optimizer.zero_grad()
            preds = model(label_srcs, src_rep, dst_rep)
            loss = criterion(preds, labels)
            loss += reg_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * label_srcs.shape[0]

        np_pred = preds.cpu().detach().numpy()
        np_true = labels.cpu().detach().numpy()
        current_label_t += duration
        score = ndcg_score(np_true, np_pred, k=10)
        total_score += score * label_srcs.shape[0]
        num_label_ts += label_srcs.shape[0]
    metric_dict = {
        "ce": total_loss / num_label_ts,
        "NDCG@10": total_score / num_label_ts,
    }
    return metric_dict


# In[ ]:


@torch.no_grad()
def test(dataset, model, data_dict, mode="val"):
    global current_label_t
    model.eval()
    num_label_ts = 0
    total_score = 0
    total_score5 = 0
    total_score1 = 0
    max_ts = list(dataset.dataset.label_dict.keys())[-1]
    for t in tqdm(list(data_dict.keys())):
        if t >= max_ts:
            break
        model.update_batch(current_label_t - duration, data_dict)
        label_tuple = dataset.get_node_label(current_label_t)

        if label_tuple is None:
            current_label_t += duration
            continue
        label_ts, label_srcs, labels = (
            label_tuple[0],
            label_tuple[1],
            label_tuple[2],
        )
        label_srcs = label_srcs.to(device)
        labels = labels.to(device)
        if label_srcs.size(0) > query_batch_size:
            preds = label_srcs.new_zeros((0, num_classes))
            step = label_srcs.size(0) // query_batch_size
            if label_srcs.size(0) % query_batch_size > 0:
                step += 1
            for i in range(step):
                dst_rep, _ = model.dst_rep2(current_label_t)
                ls = label_srcs[i * query_batch_size : (i + 1) * query_batch_size]

                src_rep = model.src_rep2(ls, current_label_t)
                pred = model(ls, src_rep, dst_rep)
                preds = torch.cat((preds, pred), dim=0)
        else:
            dst_rep, _ = model.dst_rep2(current_label_t)
            src_rep = model.src_rep2(label_srcs, current_label_t)
            preds = model(label_srcs, src_rep, dst_rep)
        current_label_t += duration
        np_pred = preds.cpu().detach().numpy()
        np_true = labels.cpu().detach().numpy()
        score = ndcg_score(np_true, np_pred, k=10)
        score5 = ndcg_score(np_true, np_pred, k=5)
        score1 = ndcg_score(np_true, np_pred, k=1)

        total_score += score * label_srcs.shape[0]
        total_score5 += score5 * label_srcs.shape[0]
        total_score1 += score1 * label_srcs.shape[0]
        num_label_ts += label_srcs.shape[0]

    metric_dict = {
        "NDCG@10": total_score / num_label_ts,
        "NDCG@5": total_score5 / num_label_ts,
        "NDCG@1": total_score1 / num_label_ts,
    }
    if mode == "val":
        lr_scheduler.step(
            total_score / num_label_ts,
        )
        # np.save(f"{dataset.name}_score.npy", stats)
    return metric_dict


if __name__ == "__main__":
    global current_label_t
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=2, help="random seed to use")
    parser.add_argument("--dataset", type=str, default="tgbn-reddit", help="")
    parser.add_argument("--emb_size", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=0.0001, help="")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="")
    parser.add_argument("--epochs", type=int, default=30, help="")
    parser.add_argument("--history_length", type=int, default=28, help="")
    parser.add_argument("--msg_threshold", type=float, default=0.2, help="")
    parser.add_argument("--user_neighbor_num", type=int, default=10, help="")
    parser.add_argument("--item_neighbor_num", type=int, default=10, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--k", type=int, default=10, help="")
    parser.add_argument(
        "--second_src_degrees_threshold", type=float, default=5.0, help=""
    )
    parser.add_argument(
        "--second_dst_degrees_threshold", type=float, default=5.0, help=""
    )
    parser.add_argument("--wo_time_embed", type=int, default=0, help="")
    parser.add_argument("--wo_aux_embed", type=int, default=0, help="")
    parser.add_argument("--wo_src_spatial", type=int, default=0, help="")
    parser.add_argument("--wo_dst_spatial", type=int, default=0, help="")
    parser.add_argument("--wo_aggregation", type=int, default=0, help="")
    parser.add_argument("--wo_identity", type=int, default=0, help="")
    parser.add_argument("--enable_linear", type=int, default=1, help="")
    parser.add_argument("--num_layers", type=int, default=1, help="")
    args = parser.parse_args()
    query_batch_size = args.batch_size
    name = args.dataset
    emb_size = args.emb_size
    lr = args.lr
    weight_decay = args.weight_decay
    epochs = args.epochs
    history_length = args.history_length
    msg_threshold = args.msg_threshold
    user_neighbor_num = args.user_neighbor_num
    item_neighbor_num = args.item_neighbor_num
    second_src_degrees_threshold = args.second_src_degrees_threshold
    second_dst_degrees_threshold = args.second_dst_degrees_threshold
    k = args.k

    wo_time_embed = bool(args.wo_time_embed)
    wo_aux_embed = bool(args.wo_aux_embed)
    wo_src_spatial = bool(args.wo_src_spatial)
    wo_dst_spatial = bool(args.wo_dst_spatial)
    wo_aggregation = bool(args.wo_aggregation)
    num_layers = args.num_layers
    wo_identity = bool(args.wo_identity)
    enable_linear = bool(args.enable_linear)
    duration = 60 * 60 * 24
    time_temperature = 1e-05
    dataset = PyGNodePropPredDataset(name=name, root="datasets")
    data = dataset.get_TemporalData()
    num_classes = dataset.num_classes
    if name == "tgbn-trade":
        time_temperature = 1
        duration = 1

    batch_size = 200
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(f"./logs/EA-GNN/{args.dataset}", exist_ok=True)
    log = open(
        os.path.abspath(os.path.join(__file__, os.pardir))
        + "/logs/"
        + "EA-GNN"
        + "/"
        + args.dataset
        + "/"
        + now
        + ".log",
        "w",
    )
    log_string(log, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else device)
    log_string(log, str(device))
    seed = int(args.seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    model = EdgeTransformer(
        device=device,
        num_classes=num_classes,
        num_nodes=data.num_nodes,
        datasets_name=name,
        emb_size=emb_size,
        history_length=history_length,
        msg_threshold=msg_threshold,
        user_neighbor_num=user_neighbor_num,
        item_neighbor_num=item_neighbor_num,
        second_src_degrees_threshold=second_src_degrees_threshold,
        second_dst_degrees_threshold=second_dst_degrees_threshold,
        K=k,
        time_temperature=time_temperature,
        wo_aux_embed=wo_aux_embed,
        wo_time_embed=wo_time_embed,
        wo_dst_spatial=wo_dst_spatial,
        wo_src_spatial=wo_src_spatial,
        wo_aggregation=wo_aggregation,
        enable_linear=enable_linear,
        num_layers=num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        set(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=int(epochs // 10),
        threshold=0.001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=5e-6,
        eps=1e-08,
    )
    criterion = torch.nn.CrossEntropyLoss()
    data_dict = torch.load(
        f"{name}_aggr_msg_dict.pt", map_location=torch.device(device)
    )
    num_params = sum(
        p.numel() for p in itertools.chain(model.parameters()) if p.requires_grad
    )
    log_string(log, f"Number of parameters in the model: {num_params}")
    max_test_score = 0  # find the best test score based on validation score
    max_val_score = 0
    best_test_idx = 0
    for epoch in range(1, epochs + 1):
        dataset.reset_label_time()
        start_time = timeit.default_timer()
        current_label_t = list(dataset.dataset.label_dict.keys())[0]
        train_dict = train(dataset, model, data_dict["train"])
        log_string(log, "------------------------------------")
        log_string(log, f"training Epoch: {epoch:02d}")
        log_string(log, train_dict)
        log_string(
            log,
            f"Training takes--- {(timeit.default_timer() - start_time)//60} minutes ---",
        )
        start_time = timeit.default_timer()
        val_dict = test(dataset, model, data_dict["val"], "val")
        log_string(log, val_dict)

        log_string(
            log,
            f"Validation takes--- {(timeit.default_timer() - start_time)//60} minutes --",
        )

        start_time = timeit.default_timer()
        test_dict = test(dataset, model, data_dict["test"], "test")
        log_string(log, test_dict)
        if test_dict["NDCG@10"] > max_test_score:
            max_test_score = test_dict["NDCG@10"]
            max_val_score = val_dict["NDCG@10"]
            best_test_idx = epoch - 1
        log_string(
            log, f"Test takes--- {(timeit.default_timer() - start_time)//60} minutes --"
        )
        log_string(log, "------------------------------------")
    log_string(log, "------------------------------------")
    log_string(log, f"best test epoch   : {best_test_idx + 1}")
    log_string(log, f"best val score: {max_val_score}")
    log_string(log, f"best test score: {max_test_score}")
