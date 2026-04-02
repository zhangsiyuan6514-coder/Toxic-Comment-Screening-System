from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ToxicCommentDataset, load_split_csv
from models.textcnn_classifier import TextCNNClassifier
from training.metrics import evaluate_model
from utils.io import read_json
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TextCNN toxic comment classifier.")
    parser.add_argument("--train-csv", type=str, default="data/processed/train_split.csv")
    parser.add_argument("--val-csv", type=str, default="data/processed/val_split.csv")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    parser.add_argument("--out", type=str, default="data/processed/textcnn_best.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    vocab_meta = read_json(args.vocab)
    token2idx: dict[str, int] = vocab_meta["token2idx"]
    max_len = int(vocab_meta["max_len"])
    pad_idx = int(vocab_meta["pad_idx"])

    train_texts, train_labels = load_split_csv(args.train_csv)
    val_texts, val_labels = load_split_csv(args.val_csv)

    train_ds = ToxicCommentDataset(train_texts, train_labels, token2idx, max_len)
    val_ds = ToxicCommentDataset(val_texts, val_labels, token2idx, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    vocab_size = len(token2idx)
    embed_dim = 128
    num_filters = 100
    filter_sizes = (3, 4, 5)
    dropout = 0.5

    model = TextCNNClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        num_classes=2,
        dropout=dropout,
        padding_idx=pad_idx,
    ).to(device)

    y_arr = np.array(train_labels, dtype=int)
    n0 = int(np.sum(y_arr == 0))
    n1 = int(np.sum(y_arr == 1))
    w1 = (n0 / max(n1, 1)) ** 0.5
    class_weights = torch.tensor([1.0, min(w1, 10.0)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1

        val_m = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={avg_loss:.4f}  "
            f"val_acc={val_m['accuracy']:.4f}  val_f1={val_m['f1']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "num_filters": num_filters,
                    "filter_sizes": list(filter_sizes),
                    "num_classes": 2,
                    "dropout": dropout,
                },
                out_path,
            )
            print(f"  saved best checkpoint (val_f1={best_f1:.4f}) -> {out_path}")

    print("done.")


if __name__ == "__main__":
    main()
