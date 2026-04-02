from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.textcnn_classifier import TextCNNClassifier
from utils.checkpoint import load_torch
from utils.io import read_json


class TextCNNInference:
    def __init__(
        self,
        vocab_path: str | Path,
        ckpt_path: str | Path,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        vocab_path = Path(vocab_path)
        ckpt_path = Path(ckpt_path)

        meta = read_json(vocab_path)
        self.token2idx: dict[str, int] = meta["token2idx"]
        self.max_len: int = int(meta["max_len"])
        self.pad_idx: int = int(meta["pad_idx"])
        self.unk_idx: int = int(meta["unk_idx"])

        ckpt = load_torch(ckpt_path, map_location=self.device)
        state = ckpt["state_dict"]
        self.model = TextCNNClassifier(
            vocab_size=int(ckpt["vocab_size"]),
            embed_dim=int(ckpt["embed_dim"]),
            num_filters=int(ckpt["num_filters"]),
            filter_sizes=tuple(int(x) for x in ckpt["filter_sizes"]),
            num_classes=int(ckpt["num_classes"]),
            dropout=float(ckpt["dropout"]),
            padding_idx=self.pad_idx,
        )
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        from data.preprocess import tokenize

        ids: list[int] = []
        for tok in tokenize(text):
            ids.append(self.token2idx.get(tok, self.unk_idx))
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        while len(ids) < self.max_len:
            ids.append(self.pad_idx)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def predict_proba_risky(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        batch = torch.stack([self.encode(t) for t in texts], dim=0)
        logits = self.model(batch)
        proba = F.softmax(logits, dim=-1)[:, 1]
        return [float(x) for x in proba.cpu()]

    def predict_proba_risky_one(self, text: str) -> float:
        return self.predict_proba_risky([text])[0]
