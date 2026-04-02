from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.build_vocab import PAD_TOKEN, UNK_TOKEN
from data.preprocess import tokenize


class ToxicCommentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        token2idx: dict[str, int],
        max_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.token2idx = token2idx
        self.max_len = max_len
        self.pad_idx = token2idx[PAD_TOKEN]
        self.unk_idx = token2idx[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.texts)

    def encode(self, text: str) -> torch.Tensor:
        ids: list[int] = []
        for tok in tokenize(text):
            ids.append(self.token2idx.get(tok, self.unk_idx))
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        while len(ids) < self.max_len:
            ids.append(self.pad_idx)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encode(self.texts[i])
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


def load_split_csv(path: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    texts = df["comment_text"].astype(str).tolist()
    labels = df["true"].astype(int).tolist()
    return texts, labels
