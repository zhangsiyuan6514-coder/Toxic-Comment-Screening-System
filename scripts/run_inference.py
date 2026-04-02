"""Batch TextCNN inference on a CSV with comment_text. Run: python scripts/run_inference.py"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pandas as pd

from models.model_inference import TextCNNInference


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/processed/val_split.csv")
    p.add_argument("--output", type=str, default="data/processed/inference_out.csv")
    p.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    p.add_argument("--ckpt", type=str, default="data/processed/textcnn_best.pt")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    infer = TextCNNInference(
        args.vocab,
        args.ckpt,
        device=args.device or None,
    )
    df = pd.read_csv(args.input)
    texts = df["comment_text"].astype(str).tolist()
    probas: list[float] = []
    for i in range(0, len(texts), args.batch):
        probas.extend(infer.predict_proba_risky(texts[i : i + args.batch]))

    df["proba_risky"] = probas
    df["pred"] = (pd.Series(probas) >= args.threshold).astype(int)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
