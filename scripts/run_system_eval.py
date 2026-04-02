"""Evaluate rules + TextCNN system on val.csv. Run: python scripts/run_system_eval.py"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pandas as pd

from routing.decision_router import DecisionRouter
from training.metrics import compute_system_metrics, print_system_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-csv", type=str, default="data/processed/val.csv")
    p.add_argument("--vocab", type=str, default="data/processed/vocab.json")
    p.add_argument("--ckpt", type=str, default="data/processed/textcnn_best.pt")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--report-csv", type=str, default="data/processed/system_eval.csv")
    args = p.parse_args()

    df = pd.read_csv(args.val_csv)
    X = df["comment_text"].astype(str)
    y_true = df["true"].astype(int).tolist()

    router = DecisionRouter(
        vocab_path=args.vocab,
        ckpt_path=args.ckpt,
        threshold=args.threshold,
        device=args.device or None,
    )

    y_pred: list[int] = []
    reasons: list[str] = []
    probas: list[float] = []
    for text in X:
        out = router.route(text)
        y_pred.append(out.pred)
        reasons.append(out.reason)
        probas.append(out.proba if out.proba is not None else -1.0)

    m = compute_system_metrics(y_true=y_true, y_pred=y_pred, reasons=reasons)
    print_system_report(m)

    df_out = df.copy()
    df_out["sys_pred"] = y_pred
    df_out["reason"] = reasons
    df_out["proba_risky"] = probas
    report_path = Path(args.report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(report_path, index=False)
    print(f"\nwrote {report_path}")

    errors = df_out[df_out["true"] != df_out["sys_pred"]].head(10)
    print("\n=== Top 10 errors (system) ===")
    for _, row in errors.iterrows():
        print("[comment]:", row["comment_text"][:200], "..." if len(str(row["comment_text"])) > 200 else "")
        print(
            "true:",
            int(row["true"]),
            "; pred:",
            int(row["sys_pred"]),
            "; reason:",
            row["reason"],
            "; proba:",
            row["proba_risky"],
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
