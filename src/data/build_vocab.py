from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from data.preprocess import tokenize
from utils.io import write_json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_token2idx(
    texts: Iterable[str],
    min_freq: int = 2,
    max_size: int = 20000,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))

    token2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, c in counter.most_common():
        if c < min_freq:
            break
        if tok in token2idx:
            continue
        if len(token2idx) >= max_size:
            break
        token2idx[tok] = len(token2idx)
    return token2idx


def save_vocab(token2idx: dict[str, int], path: str | Path, max_len: int = 256) -> None:
    payload = {
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_idx": token2idx[PAD_TOKEN],
        "unk_idx": token2idx[UNK_TOKEN],
        "max_len": max_len,
        "token2idx": token2idx,
    }
    write_json(payload, path)


def load_token2idx(path: str | Path) -> tuple[dict[str, int], int]:
    from utils.io import read_json

    data = read_json(path)
    return data["token2idx"], int(data["max_len"])
