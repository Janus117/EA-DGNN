# EA-DGNN for Dynamic Node Affinity Prediction


### Scripts
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
--neighbor_num 10 \
```