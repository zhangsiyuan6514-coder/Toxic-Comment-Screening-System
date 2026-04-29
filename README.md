# MAME — Malicious Assessment & Moderation Engine

End-to-end screening for user-generated comments: **deterministic rules** for explainability, a **router** for traffic control, and a **TextCNN** for semantic risk scoring when rules do not decide.

> **Important:** This repository does **not** include the `data/` directory. The dataset and generated model artifacts are intentionally excluded from GitHub because they can be large and may be subject to external dataset terms. See [Data setup](#data-setup) for where to place the files locally.

## Overview

Large platforms cannot manually review every comment. MAME reduces moderator load by automatically labeling comments as safe vs risky, while keeping an audit trail for rule-based decisions that humans can understand.

## What counts as “risky”?

Training labels come from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) train split.

A comment is **risky** (`true = 1`) if **any** of these columns is positive:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Otherwise, it is **safe** (`true = 0`).

At runtime, the router reports `ALLOW` / `BLOCK`, which correspond to safe / risky for the positive class used in metrics.

## System architecture

1. **Rule layer (`src/rules/rule_filter.py`)**  
   Uses hard-block keywords and hard-allow regexes. Outputs `BLOCK`, `ALLOW`, or `PASS`.

2. **Router (`src/routing/decision_router.py`)**  
   If `BLOCK` → risky immediately.  
   If `ALLOW` → safe immediately.  
   If `PASS` → run TextCNN and threshold `P(risky)`.

3. **TextCNN (`src/models/textcnn_classifier.py`)**  
   Uses word embeddings + multi-kernel CNN. Loaded through `src/models/model_inference.py` for inference.

This separates **explainable policy** from **semantic generalization**.

## Repository layout

```text
MAME/
├── src/
│   ├── data/                   # preprocess, vocab, PyTorch dataset
│   ├── models/                 # TextCNN + inference wrapper
│   ├── rules/                  # rule filter
│   ├── routing/                # decision router
│   ├── training/               # train, evaluate, metrics
│   └── utils/
├── scripts/                    # CLI entry points
├── requirements.txt
├── .gitignore
└── README.md
```

The local `data/` directory should exist on your own machine, but it is **not uploaded to GitHub**:

```text
MAME/
└── data/
    ├── raw/                    # place Kaggle train.csv here
    └── processed/              # generated splits, vocab, checkpoints, reports
```

## Data setup

### 1. Download the dataset

Download `train.csv` from the Kaggle Jigsaw Toxic Comment Classification Challenge:

<https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge>

### 2. Create the local data folders

From the repository root:

```bash
mkdir -p data/raw data/processed
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/raw, data/processed
```

### 3. Place the dataset locally

Put the downloaded file here:

```text
data/raw/train.csv
```

The project expects this local path when preparing data and training the model.

## Files intentionally excluded from GitHub

The following files are generated locally and should not be committed:

```text
data/raw/train.csv
data/processed/train_split.csv
data/processed/val_split.csv
data/processed/val.csv
data/processed/vocab.json
data/processed/textcnn_best.pt
data/processed/system_eval.csv
data/processed/inference_out.csv
```

Older TF–IDF + logistic regression artifacts may also exist from earlier experiments and should remain local:

```text
data/processed/*.joblib
```

Recommended `.gitignore` entries:

```gitignore
# Local datasets and generated ML artifacts
data/

# Python cache
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/

# Jupyter checkpoints
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db
```

## Setup

```bash
# =========================
# Install PyTorch
# =========================

# Option 1: CPU user
python -m pip install torch

# Option 2: GPU user, NVIDIA stable CUDA build
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# Option 3: GPU, newer NVIDIA GPUs that require nightly support
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# =========================
# Verify installation
# =========================

python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# =========================
# Install remaining dependencies
# =========================

python -m pip install -r requirements.txt
```

## Quickstart

Run all commands from the repository root.

| Step | Command | Purpose |
|---|---|---|
| 1 | `python scripts/prepare_data.py` | Build train/validation splits and `vocab.json` |
| 2 | `python scripts/run_train.py` | Train TextCNN and save `textcnn_best.pt` |
| 3 | `python scripts/run_evaluate.py` | Evaluate model-only performance on `val_split.csv` |
| 4 | `python scripts/run_system_eval.py` | Evaluate rules + router + TextCNN and write `system_eval.csv` |

Optional batch scoring on any CSV with a `comment_text` column:

```bash
python scripts/run_inference.py --input data/processed/val_split.csv --output data/processed/inference_out.csv
```

## Common CLI flags

Training:

```bash
python scripts/run_train.py --epochs 5 --batch-size 64 --lr 1e-3
```

System evaluation:

```bash
python scripts/run_system_eval.py --threshold 0.5 --report-csv data/processed/system_eval.csv
```

All scripts change the working directory to the repository root so relative paths such as `data/processed/...` resolve correctly.

## Evaluation

`src/training/metrics.py` computes binary accuracy, precision, recall, and F1 for the model.

System-level evaluation also reports rule/model coverage so you can see how many samples were decided by:

- `rule:block`
- `rule:allow`
- `model`

`run_system_eval.py` prints a confusion matrix for the positive class, where:

```text
positive class = risky
```

It also shows sample errors for debugging.

## GitHub cleanup note

If `data/` was already committed, remove it from Git tracking while keeping the local files:

```bash
git rm -r --cached data
git add .gitignore README.md
git commit -m "Stop tracking local data artifacts"
git push
```

This removes `data/` from the current GitHub repository state but keeps your local `data/` folder.

If large files were committed in older history and GitHub still reports the repository as too large, you may need to rewrite Git history with a tool such as `git filter-repo` or BFG Repo-Cleaner. Only do that after backing up the repository, because it changes commit history.

## Limitations and future work

- Rules are intentionally small demos; production systems need curated keyword lists, locale-specific policies, and regular audits.
- TextCNN uses simple whitespace tokenization and a fixed `max_len`; stronger tokenization or transformer-based models may improve robustness.
- Class imbalance still affects the precision/recall trade-off; threshold tuning should match product policy.
- The current repository is designed for reproducible local training, not for hosting datasets or checkpoints directly on GitHub.
