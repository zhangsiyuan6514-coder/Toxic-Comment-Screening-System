# MAME — Malicious Assessment & Moderation Engine

End-to-end screening for user-generated comments: **deterministic rules** for explainability, a **router** for traffic control, and a **TextCNN** for semantic risk scoring when rules do not decide.

## Overview

Large platforms cannot manually review every comment. MAME reduces moderator load by automatically labeling comments as safe vs risky, while keeping an audit trail for rule-based decisions that humans can understand. 

## What counts as “risky”?

Training labels come from the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) train split. A comment is **risky** (`true = 1`) if **any** of these columns is positive: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Otherwise it is **safe** (`true = 0`).

At runtime the router reports `ALLOW` / `BLOCK`, which correspond to safe / risky for the positive class used in metrics.

## System architecture

1. **Rule layer (`src/rules/rule_filter.py`)** — `Hard block` keywords and `hard allow` regexes. Outputs `BLOCK`, `ALLOW`, or `PASS`.
2. **Router (`src/routing/decision_router.py`)** — If `BLOCK` → risky immediately; if `ALLOW` → safe immediately; if `PASS` → run TextCNN and threshold `P(risky)`.
3. **TextCNN (`src/models/textcnn_classifier.py`)** — Word embeddings + multi-kernel CNN; loaded via `src/models/model_inference.py` for inference.

This separates **explainable policy** (rules) from **semantic generalization** (neural model).

## Repository layout

```
MAME/
├── data/
│   ├── raw/                    # place Kaggle train.csv (and optional test files) here
│   └── processed/              # splits, vocab.json, textcnn_best.pt, system_eval.csv, ...
├── src/
│   ├── data/                   # preprocess, vocab, PyTorch dataset
│   ├── models/                 # TextCNN + inference wrapper
│   ├── rules/                  # rule filter
│   ├── routing/                # decision router
│   ├── training/               # train, evaluate, metrics
│   └── utils/
├── scripts/                    # CLI entry points
├── requirements.txt
└── README.md
```

Generated artifacts (after running the pipeline) include:

- `train_split.csv`, `val_split.csv`, `val.csv` — stratified 90/10 split with `comment_text`, `true`, and `id`
- `vocab.json` — token vocabulary built **only from** the training split
- `textcnn_best.pt` — checkpoint chosen by best validation F1 during training
- `system_eval.csv` — per-row system predictions from `run_system_eval.py`

Older **TF–IDF + logistic regression** `*.joblib` files may still exist under `data/processed/` from earlier experiments; the live router uses **only** TextCNN plus rules.

## Setup

```bash
# =========================
# Install PyTorch
# =========================

# Option 1: CPU user
python -m pip install torch

# Option 2: GPU user (NVIDIA, standard GPUs)
# Use this if your GPU is supported by stable PyTorch (e.g., RTX 20/30/40 series)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# Option 3: GPU (new NVIDIA GPUs, e.g., RTX 50 series)
# Required for GPUs with compute capability sm_120 (e.g., RTX 5070)
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128


# =========================
# Verify installation
# =========================

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Output Example
True
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
NVIDIA GeForce RTX 5070 Laptop GPU

# =========================
# Install remaining dependencies
# =========================

python -m pip install -r requirements.txt
```

Place `train.csv` from the Jigsaw competition in `data/raw/`.

## Quickstart (run from repository root)

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python scripts/prepare_data.py` | Build splits + `vocab.json` |
| 2 | `python scripts/run_train.py` | Train TextCNN; saves `textcnn_best.pt` |
| 3 | `python scripts/run_evaluate.py` | Evaluate **model only** on `val_split.csv` |
| 4 | `python scripts/run_system_eval.py` | Evaluate **rules + router + TextCNN**; writes `system_eval.csv` |

Optional batch scoring on any CSV with a `comment_text` column:

```bash
python scripts/run_inference.py --input data/processed/val_split.csv --output data/processed/inference_out.csv
```

### Common CLI flags

- **Training**: `scripts/run_train.py --epochs 5 --batch-size 64 --lr 1e-3` (defaults are set in `src/training/train.py`).
- **System eval**: `scripts/run_system_eval.py --threshold 0.5 --report-csv data/processed/system_eval.csv`.

All scripts change the working directory to the repo root so relative paths like `data/processed/...` resolve correctly.

## Evaluation

- **`training/metrics.py`** — Binary accuracy / precision / recall / F1 for the model; **system** metrics add rule vs model coverage (`rule:` reasons vs `model`).
- **`run_system_eval.py`** prints a confusion matrix for the **positive class = risky** and shows sample errors.

## Limitations and future work

- Rules are intentionally small demos; production systems need curated lists, locales, and regular audits.
- TextCNN uses simple whitespace tokenization and a fixed `max_len`; other tokenizers or architectures may improve robustness.
- Class imbalance (many safe / few toxic comments) still affects precision/recall trade-offs; threshold tuning should match product policy.
