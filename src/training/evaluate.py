from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import ToxicCommentDataset, load_split_csv
from models.textcnn_classifier import TextCNNClassifier
from training.metrics import evaluate_model
from utils.checkpoint import load_torch
from utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TextCNN on a labeled CSV split.")
    parser.add_argument("--val-csv", type=str, default="data/processed/val_split.csv")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    parser.add_argument("--ckpt", type=str, default="data/processed/textcnn_best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    vocab_meta = read_json(args.vocab)
    token2idx: dict[str, int] = vocab_meta["token2idx"]
    max_len = int(vocab_meta["max_len"])
    pad_idx = int(vocab_meta["pad_idx"])

    texts, labels = load_split_csv(args.val_csv)
    ds = ToxicCommentDataset(texts, labels, token2idx, max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = load_torch(args.ckpt, map_location=device)
    model = TextCNNClassifier(
        vocab_size=int(ckpt["vocab_size"]),
        embed_dim=int(ckpt["embed_dim"]),
        num_filters=int(ckpt["num_filters"]),
        filter_sizes=tuple(int(x) for x in ckpt["filter_sizes"]),
        num_classes=int(ckpt["num_classes"]),
        dropout=float(ckpt["dropout"]),
        padding_idx=pad_idx,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    m = evaluate_model(model, loader, device)
    print("=== TextCNN validation metrics ===")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k}: {m[k]:.4f}")
    print(f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


if __name__ == "__main__":
    main()
