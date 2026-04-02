from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_torch(path: str | Path, map_location: str | torch.device | None = None) -> Any:
    path = Path(path)
    kwargs: dict = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
