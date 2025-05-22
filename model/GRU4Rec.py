import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .BaseModule import TimeEncode


class GRU4Rec(nn.Module):
    def __init__(
        self,
        device,
        num_items,
        num_nodes,
        datasets_name,
        history_length,
        embedding_size=64,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
        pad_idx=0,
    ):
        """
        Parameters:
            num_items (int): Total number of items (including padding index).
            embedding_size (int): Dimension of the item embeddings.
            hidden_size (int): Dimension of GRU hidden state.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout probability on the embedding.
            pad_idx (int): Index used for padding.
        """
        super(GRU4Rec, self).__init__()
        # Input embedding layer; using pad_idx to ignore padding tokens.
        self.device = device
        self.num_nodes = num_nodes
        self.num_classes = num_items
        self.datasets_name = datasets_name
        self.msg_len = 2 if self.datasets_name == "tgbn-reddit" else 1
        self.offset = 0 if datasets_name == "tgbn-trade" else num_items
        self.history_length = history_length
        self.embedding = nn.Embedding(num_items, embedding_size)
        self.dropout = nn.Dropout(dropout)
        # GRU: batch_first=True so that input shape is (batch, seq_len, embedding_size)
        self.gru = nn.GRU(
            embedding_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.time_encoder = TimeEncode(embedding_size)
        self.w = nn.Linear(self.msg_len, embedding_size)
        # A linear layer to project from hidden state space to embedding space.
        self.fc = nn.Linear(hidden_size, num_items)
        self.src_history = None
        self.src_dst_history = None
        self.src_history_timestamp = None
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()

    def reset_state(self):
        self.src_history = (
            torch.zeros(
                (
                    self.num_nodes - self.offset,
                    self.history_length,
                    self.msg_len,
                )
            )
            .float()
            .to(self.device)
        )
        self.src_dst_history = (
            torch.zeros(
                (
                    self.num_nodes - self.offset,
                    self.history_length,
                )
            )
            .int()
            .to(self.device)
        )
        self.src_history_timestamp = (
            torch.zeros(
                (
                    self.num_nodes - self.offset,
                    self.history_length,
                )
            )
            .int()
            .to(self.device)
        )

    @torch.no_grad()
    def update_batch(self, src, dst, ts, msg, aux_msg=None):
        if src.shape[0] == 0:
            return
        src = src.cpu().numpy().tolist()
        msg = msg.cpu().numpy().tolist()
        dst = dst.cpu().numpy().tolist()

        for s, d, m in zip(src, dst, msg):
            s = s - self.offset
            self.src_history[s] = torch.roll(self.src_history[s], shifts=1, dims=-2)
            self.src_history[s, 0, :] = m
            self.src_history_timestamp[s] = torch.roll(
                self.src_history_timestamp[s], shifts=1, dims=-1
            )
            self.src_history_timestamp[s, 0] = ts
            self.src_dst_history[s] = torch.roll(
                self.src_dst_history[s], shifts=1, dims=-1
            )
            self.src_dst_history[s, 0] = d

    def forward(self, label_srcs):

        valid_seq = (self.src_history[label_srcs - self.offset][:, :, 0] > 0).sum(
            dim=-1
        )
        # Replace any 0 in seq_lengths with 1
        valid_seq = torch.clamp(valid_seq, min=1)

        input_sequences = self.src_dst_history[label_srcs - self.offset].flatten()
        emb = self.embedding(input_sequences)
        emb = emb.reshape(label_srcs.shape[0], self.history_length, -1)
        emb = self.dropout(emb)
        emb = (
            emb
            + self.w(self.src_history[label_srcs - self.offset])
            + self.time_encoder(
                self.src_history_timestamp[label_srcs - self.offset].float()
            ).reshape(label_srcs.shape[0], self.history_length, -1)
        )

        # Pack the sequence for variable-length processing.
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, valid_seq.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        # Unpack the output: (batch, seq_len, hidden_size)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        batch_size = input_sequences.size(0)
        # Gather the outputs at the last valid index for each sequence.
        idx = (valid_seq - 1).view(-1, 1, 1).expand(valid_seq.size(0), 1, output.size(2))
        last_outputs = output.gather(1, idx).squeeze(1)

        # Project the last GRU output to the embedding space.
        pred_embedding = self.fc(last_outputs)  # (batch, embedding_size)
        return pred_embedding
