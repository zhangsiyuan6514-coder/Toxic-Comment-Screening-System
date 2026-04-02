"""
Build stratified train/val splits, val.csv, and vocab.json from data/raw/train.csv.
Run from repo root: python scripts/prepare_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pandas as pd
from sklearn.model_selection import train_test_split

from data.build_vocab import build_token2idx, save_vocab
from utils.io import ensure_dir

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def main() -> None:
    raw_path = ROOT / "data" / "raw" / "train.csv"
    out_dir = ensure_dir(ROOT / "data" / "processed")

    df = pd.read_csv(raw_path)
    for c in LABEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    y = (df[LABEL_COLS].max(axis=1) >= 0.5).astype(int)

    out = pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "comment_text": df["comment_text"].astype(str),
            "true": y,
        }
    )

    train_df, val_df = train_test_split(
        out, test_size=0.1, random_state=42, stratify=y
    )

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)

    token2idx = build_token2idx(
        train_df["comment_text"].tolist(),
        min_freq=2,
        max_size=20000,
    )
    save_vocab(token2idx, out_dir / "vocab.json", max_len=256)

    print(f"train={len(train_df)} val={len(val_df)} vocab={len(token2idx)}")
    print(f"wrote {out_dir / 'train_split.csv'}, val_split.csv, val.csv, vocab.json")


if __name__ == "__main__":
    main()
