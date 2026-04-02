from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNClassifier(nn.Module):
    """Multi-filter CNN over token embeddings (Kim-style)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_classes: int = 2,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in filter_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        emb = self.embedding(x)  # [B, L, E]
        emb = emb.transpose(1, 2)  # [B, E, L]
        pooled: list[torch.Tensor] = []
        for conv in self.convs:
            h = F.relu(conv(emb))  # [B, F, L-k+1]
            p = F.max_pool1d(h, kernel_size=h.size(2))  # [B, F, 1]
            pooled.append(p.squeeze(2))
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)
