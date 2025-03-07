# EA-DGNN for Dynamic Node Affinity Prediction

## Requirements
python 3.10 

torch 2.1

torch_geometric

torch_scatter 

torch_sparse
## Preprocessed Dataset
Genre dataset can be found in https://drive.google.com/file/d/1rMXBPWeWlvJ7SH-A_ZSD6k-5xHvxAueh/view?usp=drive_link

Reddit dataset can be found in https://drive.google.com/file/d/1do5sEz394inv6H6kcsanCFV2xO2BNQ1G/view?usp=drive_link

Token dataset can be found in https://drive.google.com/file/d/1Awcb9U9e9Rs6CNntafstfzp4nN6Htr6_/view?usp=drive_link


## Installation

Follow these steps to set up the project:

## Scripts
* Example of run EA_DGNN on the dynamic node affinity prediction on *genre* dataset:
```
python train_gpu.py \
--dataset tgbn-genre \
--emb_size 32  \
--msg_threshold 0.2 \
--batch_size 128 \
--lr 0.0003 \
--epochs 100 \
--second_src_degrees_threshold 5 \
--second_dst_degrees_threshold 5 \
--history_length 28 \
--user_neighbor_num 10 \
--item_neighbor_num 10
```
--dataset: Name of the dataset to use (tgbn-trade, tgbn-genre, tgbn-reddit, or tgbn-token).

--emb_size: Dimension of node embeddings (e.g., 32).

--msg_threshold: Message passing threshold.

--batch_size: Batch size for training.

--lr: Learning rate.

--epochs: Number of training epochs.

--second_src_degrees_threshold: Degree threshold for second-order source nodes.

--second_dst_degrees_threshold: Degree threshold for second-order destination nodes.

--history_length: Length of historical data to consider.

--user_neighbor_num: Number of user neighbors to sample.

--item_neighbor_num: Number of item neighbors to sample.
