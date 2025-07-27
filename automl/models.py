import numpy as np
import torch
import torch.nn as nn


class SimpleFFNN(nn.Module):
    def __init__(self, input_dim, hidden=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x): return self.model(x)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dims: dict[str, int]):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.dropout = nn.Dropout(0.3)
        self.heads   = nn.ModuleDict({
            t: nn.Linear(2*hidden_dim, c)           # 2* for bi‑directional
            for t, c in output_dims.items()
        })

    def forward(self, task, **kwargs):
        ids      = kwargs["input_ids"].long()          # <- cast
        lengths  = kwargs["lengths"].cpu()             # safer for packing

        x = self.embed(ids)                            # embed once
        x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[0], h[1]), dim=-1)            # bi‑LSTM
        return self.heads[task](h)

    
        