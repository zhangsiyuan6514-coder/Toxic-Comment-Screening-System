from __future__ import annotations

import re


def clean_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> list[str]:
    s = clean_text(text)
    if not s:
        return []
    return s.split()
