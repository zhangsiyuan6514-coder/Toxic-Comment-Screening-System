from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.model_inference import TextCNNInference
from rules.rule_filter import RuleFilter


@dataclass
class RouterResult:
    decision: str          # "ALLOW" or "BLOCK"
    pred: int              # 0 or 1  (1 = risky)
    proba: float | None    # P(risky=1), None if decided by rules
    reason: str            # "rule:ALLOW" / "rule:BLOCK" / "model"


class DecisionRouter:
    """
    1) Rules: hard ALLOW / hard BLOCK (explainable).
    2) Otherwise TextCNN: probability of risky vs threshold.
    """

    def __init__(
        self,
        vocab_path: str | Path = "data/processed/vocab.json",
        ckpt_path: str | Path = "data/processed/textcnn_best.pt",
        threshold: float = 0.5,
        device: str | None = None,
    ):
        self.threshold = float(threshold)
        self.rule_filter = RuleFilter()
        self.model = TextCNNInference(vocab_path, ckpt_path, device=device)

    def route(self, text: str) -> RouterResult:
        text = "" if text is None else str(text)

        rule = self.rule_filter.apply(text)
        if rule == "ALLOW":
            return RouterResult(decision="ALLOW", pred=0, proba=None, reason="rule:ALLOW")
        if rule == "BLOCK":
            return RouterResult(decision="BLOCK", pred=1, proba=None, reason="rule:BLOCK")

        proba_risky = self.model.predict_proba_risky_one(text)
        pred = 1 if proba_risky >= self.threshold else 0
        decision = "BLOCK" if pred == 1 else "ALLOW"
        return RouterResult(decision=decision, pred=pred, proba=proba_risky, reason="model")
