from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = len(y_true)
    acc = _safe_div(tp + tn, n)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


@torch.inference_mode()
def evaluate_model(model, loader, device) -> dict[str, float]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.tolist())
    return binary_metrics(np.array(ys), np.array(ps))


@dataclass
class SystemMetrics:
    n: int
    accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision_pos: float
    recall_pos: float
    f1_pos: float
    ruled_pct: float
    model_pct: float


def compute_system_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    reasons: Iterable[str],
) -> SystemMetrics:
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    reasons = list(reasons)

    n = len(y_true)
    if not (n == len(y_pred) == len(reasons)):
        raise ValueError("Length mismatch")

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    acc = _safe_div(tp + tn, n)
    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    ruled = sum(1 for r in reasons if r.startswith("rule:"))
    model = n - ruled
    ruled_pct = _safe_div(ruled, n)
    model_pct = _safe_div(model, n)

    return SystemMetrics(
        n=n,
        accuracy=acc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision_pos=precision_pos,
        recall_pos=recall_pos,
        f1_pos=f1_pos,
        ruled_pct=ruled_pct,
        model_pct=model_pct,
    )


def print_system_report(m: SystemMetrics) -> None:
    print("\n=== System Metrics ===")
    print(f"N: {m.n}")
    print(f"Accuracy: {m.accuracy:.4f}")
    print(f"Confusion Matrix (pos=risky=1): TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn}")
    print(f"Precision(pos): {m.precision_pos:.4f}")
    print(f"Recall(pos):    {m.recall_pos:.4f}")
    print(f"F1(pos):        {m.f1_pos:.4f}")
    print(f"Rule coverage:  {m.ruled_pct:.2%}")
    print(f"Model coverage: {m.model_pct:.2%}")
