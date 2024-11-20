from typing import Optional, cast, Union, List, overload, Literal
from tqdm import tqdm
import numpy as np
import pandas as pd
import os.path as osp
import time
import csv
import datetime
from datetime import date


"""
functions for un_trade dataset
---------------------------------------
"""


def load_edgelist_trade(fname: str, label_size=255):
    """
    load the edgelist into pandas dataframe
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    # print("numpy allocated")
    node_ids = {}  # dictionary for node ids
    node_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = int(row[0])
                u = row[1]
                v = row[2]
                w = float(row[3])
                if u not in node_ids:
                    node_ids[u] = node_uid
                    node_uid += 1

                if v not in node_ids:
                    node_ids[v] = node_uid
                    node_uid += 1

                u = node_ids[u]
                i = node_ids[v]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([w])
                idx += 1

    return (
        pd.DataFrame(
            {"u": u_list, "i": i_list, "ts": ts_list, "idx": idx_list, "w": w_list}
        ),
        feat_l,
        node_ids,
    )


def load_trade_label_dict(
    fname: str,
    node_ids: dict,
) -> dict:
    """
    load node labels into a nested dictionary instead of pandas dataobject
    {ts: {node_id: label_vec}}
    Parameters:
        fname: str, name of the input file
        node_ids: dictionary of user names mapped to integer node ids
    Returns:
        node_label_dict: a nested dictionary of node labels
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    label_size = len(node_ids)
    # label_vec = np.zeros(label_size)

    node_label_dict = {}  # {ts: {node_id: label_vec}}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = int(row[0])
                u = node_ids[row[1]]
                v = node_ids[row[2]]
                weight = float(row[3])

                if ts not in node_label_dict:
                    node_label_dict[ts] = {u: np.zeros(label_size)}

                if u not in node_label_dict[ts]:
                    node_label_dict[ts][u] = np.zeros(label_size)

                node_label_dict[ts][u][v] = weight
                idx += 1
        return node_label_dict


"""
functions for tgbn-token
---------------------------------------
"""


def load_edgelist_token(
    fname: str,
    label_size: int = 1001,
) -> pd.DataFrame:
    """
    load the edgelist into pandas dataframe
    also outputs index for the user nodes and genre nodes
    Parameters:
        fname: str, name of the input file
        label_size: int, number of genres
    Returns:
        df: a pandas dataframe containing the edgelist data
    """
    feat_size = 2
    num_lines = sum(1 for line in open(fname)) - 1
    # print("number of lines counted", num_lines)
    print("there are ", num_lines, " lines in the raw data")
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)

    node_ids = {}
    rd_dict = {}
    node_uid = label_size  # node ids start after all the genres
    sr_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # [timestamp,user_address,token_address,value,IsSender]
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = row[0]
                src = row[1]
                token = row[2]
                w = float(row[3])
                attr = float(row[4])
                if src not in node_ids:
                    node_ids[src] = node_uid
                    node_uid += 1
                if token not in rd_dict:
                    rd_dict[token] = sr_uid
                    sr_uid += 1
                u = node_ids[src]
                i = rd_dict[token]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([w, attr])
                idx += 1

        return (
            pd.DataFrame(
                {
                    "u": u_list,
                    "i": i_list,
                    "ts": ts_list,
                    "label": label_list,
                    "idx": idx_list,
                    "w": w_list,
                }
            ),
            feat_l,
            node_ids,
            rd_dict,
        )


"""
functions for subreddits dataset
---------------------------------------
"""


def load_edgelist_sr(
    fname: str,
    label_size: int = 2221,
) -> pd.DataFrame:
    """
    load the edgelist into pandas dataframe
    also outputs index for the user nodes and genre nodes
    Parameters:
        fname: str, name of the input file
        label_size: int, number of genres
    Returns:
        df: a pandas dataframe containing the edgelist data
    """
    feat_size = 2
    num_lines = sum(1 for line in open(fname)) - 1
    # print("number of lines counted", num_lines)
    print("there are ", num_lines, " lines in the raw data")
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    label_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)

    node_ids = {}
    rd_dict = {}
    node_uid = label_size  # node ids start after all the genres
    sr_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'subreddit', 'num_words', 'score']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = row[0]
                src = row[1]
                subreddit = row[2]
                num_words = int(row[3])
                score = int(row[4])
                if src not in node_ids:
                    node_ids[src] = node_uid
                    node_uid += 1
                if subreddit not in rd_dict:
                    rd_dict[subreddit] = sr_uid
                    sr_uid += 1
                w = float(score)
                u = node_ids[src]
                i = rd_dict[subreddit]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.array([float(num_words),w, ])
                idx += 1

        return (
            pd.DataFrame(
                {
                    "u": u_list,
                    "i": i_list,
                    "ts": ts_list,
                    "label": label_list,
                    "idx": idx_list,
                    "w": w_list,
                }
            ),
            feat_l,
            node_ids,
            rd_dict,
        )


def load_labels_sr(
    fname,
    node_ids,
    rd_dict,
):
    """
    load the node labels for subreddit dataset
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    # day, user_idx, label_vec
    label_size = len(rd_dict)
    label_vec = np.zeros(label_size)
    ts_prev = 0
    prev_user = 0

    ts_list = []
    node_id_list = []
    y_list = []

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'subreddit', 'num_words', 'score']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                user_id = node_ids[int(row[1])]
                ts = int(row[0])
                sr_id = int(rd_dict[row[2]])
                weight = float(row[3])
                if idx == 1:
                    ts_prev = ts
                    prev_user = user_id
                # the next day
                if ts != ts_prev:
                    ts_list.append(ts_prev)
                    node_id_list.append(prev_user)
                    y_list.append(label_vec)
                    label_vec = np.zeros(label_size)
                    ts_prev = ts
                    prev_user = user_id
                else:
                    label_vec[sr_id] = weight

                if user_id != prev_user:
                    ts_list.append(ts_prev)
                    node_id_list.append(prev_user)
                    y_list.append(label_vec)
                    prev_user = user_id
                    label_vec = np.zeros(label_size)
                idx += 1
        return pd.DataFrame({"ts": ts_list, "node_id": node_id_list, "y": y_list})


def load_label_dict(fname: str, node_ids: dict, rd_dict: dict) -> dict:
    """
    load node labels into a nested dictionary instead of pandas dataobject
    {ts: {node_id: label_vec}}
    Parameters:
        fname: str, name of the input file
        node_ids: dictionary of user names mapped to integer node ids
        rd_dict: dictionary of subreddit names mapped to integer node ids
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    # day, user_idx, label_vec
    label_size = len(rd_dict)
    node_label_dict = {}  # {ts: {node_id: label_vec}}

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        # ['ts', 'src', 'dst', 'w']
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                u = node_ids[row[1]]
                ts = int(row[0])
                v = int(rd_dict[row[2]])
                weight = float(row[3])
                if ts not in node_label_dict:
                    node_label_dict[ts] = {u: np.zeros(label_size)}

                if u not in node_label_dict[ts]:
                    node_label_dict[ts][u] = np.zeros(label_size)

                node_label_dict[ts][u][v] = weight
                idx += 1
        return node_label_dict





"""
functions for last fm genre
-------------------------------------------
"""


def load_edgelist_datetime(fname, label_size=514):
    """
    load the edgelist into a pandas dataframe
    use numpy array instead of list for faster processing
    assume all edges are already sorted by time
    convert all time unit to unix time

    time, user_id, genre, weight
    """
    feat_size = 1
    num_lines = sum(1 for line in open(fname)) - 1
    print("number of lines counted", num_lines)
    u_list = np.zeros(num_lines)
    i_list = np.zeros(num_lines)
    ts_list = np.zeros(num_lines)
    feat_l = np.zeros((num_lines, feat_size))
    idx_list = np.zeros(num_lines)
    w_list = np.zeros(num_lines)
    #print("numpy allocated")
    node_ids = {}  # dictionary for node ids
    label_ids = {}  # dictionary for label ids
    node_uid = label_size  # node ids start after the genre nodes
    label_uid = 0

    with open(fname, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        idx = 0
        for row in tqdm(csv_reader):
            if idx == 0:
                idx += 1
            else:
                ts = int(row[0])
                user_id = row[1]
                genre = row[2]
                w = float(row[3])

                if user_id not in node_ids:
                    node_ids[user_id] = node_uid
                    node_uid += 1

                if genre not in label_ids:
                    label_ids[genre] = label_uid
                    if label_uid >= label_size:
                        print("id overlap, terminate")
                    label_uid += 1

                u = node_ids[user_id]
                i = label_ids[genre]
                u_list[idx - 1] = u
                i_list[idx - 1] = i
                ts_list[idx - 1] = ts
                idx_list[idx - 1] = idx
                w_list[idx - 1] = w
                feat_l[idx - 1] = np.asarray([w])
                idx += 1

    return (
        pd.DataFrame(
            {"u": u_list, "i": i_list, "ts": ts_list, "idx": idx_list, "w": w_list}
        ),
        feat_l,
        node_ids,
        label_ids,
    )


def load_genre_list(fname):
    """
    load the list of genres
    """
    if not osp.exists(fname):
        raise FileNotFoundError(f"File not found at {fname}")

    edgelist = open(fname, "r")
    lines = list(edgelist.readlines())
    edgelist.close()

    genre_index = {}
    ctr = 0
    for i in range(1, len(lines)):
        vals = lines[i].split(",")
        genre = vals[0]
        if genre not in genre_index:
            genre_index[genre] = ctr
            ctr += 1
        else:
            raise ValueError("duplicate in genre_index")
    return genre_index




