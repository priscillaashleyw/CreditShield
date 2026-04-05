# TabTransformer for Credit Risk Prediction

## Overview

This project implements **TabTransformer** — a deep learning architecture that applies Transformer self-attention to tabular data — for binary credit risk classification (loan default prediction) on the Lending Club dataset (2013–2014).

The pipeline is fully integrated end-to-end:
- `load_data.py` handles all data acquisition, filtering, and splitting
- `tab_transformer.py` defines the model architecture
- `train_tab_transformer.py` handles preprocessing, training, and evaluation
- `run_tab_transformer.py` is the single entry point that wires everything together

---

## Quick Start

```bash
# 1. Navigate to the project root and activate the virtual environment
cd CreditShield
source venv/bin/activate

# 2. Install dependencies
cd credit-risk-prediction-project/training
pip install -r requirements.txt
pip install torch einops pyyaml

# 3. Run the full pipeline (data loading → training → evaluation)
python run_tab_transformer.py
```

That's it. `load_data.py` will automatically download the Lending Club dataset from Kaggle if it is not already present locally (requires `~/.kaggle/kaggle.json` credentials).

---

## How the Pipeline Works

```
run_tab_transformer.py
    │
    ├── [1] Load config  ←  config/tab_transformer_config.yaml
    │
    ├── [2] src/load_data.DataLoader
    │       ├── load_and_filter_data()
    │       │     • Downloads CSV from Kaggle if not on disk
    │       │     • Loads 39 essential columns
    │       │     • Filters to 2013–2014 loans
    │       │     • Drops post-origination leakage columns
    │       │     • Applies log1p to annual_inc, revol_bal
    │       │     • Creates structural missingness indicators
    │       │       (mths_since_last_*_missing columns)
    │       ├── define_target()
    │       │     • 1 = Charged Off / Default / Late 31-120 days
    │       │     • 0 = Fully Paid
    │       └── random_split()  →  80% train / 20% test (stratified)
    │
    └── [3] src/train_tab_transformer.TabTransformerTrainer
            ├── prepare_data()
            │     • Drops date metadata columns
            │     • LabelEncoder for 7 categorical features
            │     • Median imputation + StandardScaler for 33 numerical features
            │     • Auto-captures dynamic _missing indicator columns
            │     • Builds PyTorch DataLoaders
            ├── build_model()   →  TabTransformer (tab_transformer.py)
            ├── train()
            │     • AdamW optimiser + CosineAnnealingWarmRestarts
            │     • Weighted BCELoss (pos_weight for class imbalance)
            │     • Gradient clipping (max_norm=1.0)
            │     • Threshold re-tuned every 5 epochs via PR curve
            │     • Early stopping on AUC + F1 combined score
            └── evaluate()  →  prints full metrics report
```

---

## Model Architecture

```
Input
  ├── Categorical features (7)
  │     └── LabelEncoder → Embedding (dim=32) per feature
  │                   → Stack: (batch, 7, 32)
  │                   → TransformerBlock × 4
  │                         MultiHeadAttention (heads=4)
  │                         FeedForward (GELU)
  │                         LayerNorm + Residuals
  │                   → AdaptiveAvgPool → (batch, 32)
  │
  └── Numerical features (33)
        └── StandardScaler → bypass: (batch, 33)

Fusion: concat → (batch, 65)
  └── MLP: 65 → 256 → 128 → 1
        └── Sigmoid → P(default) ∈ [0, 1]
```

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `embedding_dim` | 32 | Dimension of each categorical embedding |
| `depth` | 4 | Number of Transformer encoder blocks |
| `heads` | 4 | Attention heads (must be ≤ num categorical features) |
| `dim_head` | 32 | Dimension per attention head |
| `mlp_dim` | 256 | Hidden size in feed-forward and final MLP layers |
| `dropout` | 0.2 | Dropout rate in attention and feed-forward layers |
| `learning_rate` | 0.0005 | Initial AdamW learning rate |
| `batch_size` | 256 | Training batch size |
| `weight_decay` | 0.01 | L2 regularisation strength |
| `early_stopping_patience` | 15 | Epochs without improvement before stopping |

---

## Features Used

### Categorical (7) — embedding + transformer pathway
| Feature | Description |
|---|---|
| `term` | Loan term: "36 months" / "60 months" |
| `emp_length` | Employment length: "< 1 year" … "10+ years" |
| `home_ownership` | RENT, OWN, MORTGAGE, OTHER |
| `verification_status` | Verified, Not Verified, Source Verified |
| `purpose` | Loan purpose (debt consolidation, credit card, etc.) |
| `addr_state` | US state of borrower (50 states) |
| `initial_list_status` | Loan listing status: w / f |

### Numerical (33) — bypass pathway
Core financials (`loan_amnt`, `int_rate`, `annual_inc`, `dti`, `revol_util`, …), FICO scores (`last_fico_range_high`, `last_fico_range_low`), extended credit history (`avg_cur_bal`, `bc_util`, `mort_acc`, `tot_hi_cred_lim`, …), structural missingness columns (`mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog` filled with 999), and binary missingness indicators added dynamically by `load_data.py` (`mths_since_last_*_missing`).

---

## Class Imbalance Handling

Lending Club data is highly imbalanced (~80% good loans, ~20% defaults). Three mechanisms address this:

1. **Positive-class weight** — `pos_weight = n_neg / n_pos` applied per sample in BCELoss so defaults contribute proportionally more to the gradient
2. **Optimal threshold selection** — instead of a fixed 0.5 cutoff, the threshold is tuned every 5 epochs using the precision-recall curve to maximise F1
3. **Combined early stopping metric** — model selection uses `AUC + F1` so the checkpoint is not just a high-AUC but recall-poor model

---

## Expected Training Output

```
[1/3] Loading data ...
Loading data according to paper methodology...
✓ Loaded 421,095 rows, 39 columns
✓ Filtered to years [2013, 2014]: 235,629 rows
✓ Removed incomplete loans: 188,183 remaining

[2/3] Splitting data ...
✓ Training: 150,546 samples
✓ Testing:   37,637 samples
✓ Training default rate: 18.4%
✓ Testing default rate:  18.4%

[3/3] Training TabTransformer ...

  Numerical features  : 33
  Categorical features: 7
    term: 2 categories
    emp_length: 12 categories
    home_ownership: 5 categories
    ...

  Class imbalance ratio  : 4.43:1 (neg:pos)
  Positive-class weight  : 4.43

  TabTransformer built  : 387,713 parameters
    Categorical tokens  : 7 (embed_dim=32)
    Numerical bypass    : 33 features
    Attention heads     : 4, depth=4, dim_head=32

Starting training for up to 100 epochs (patience=15) ...
─────────────────────────────────────────────────────────────────

Epoch   1/100 │ Loss: 0.6823 │ LR: 5.00e-04
  AUC: 0.6912 │ F1: 0.3845 │ Prec: 0.4201 │ Rec: 0.3543
  Threshold: 0.500 │ TP: 2,441 │ FP: 3,372
  ✓ Best model saved (AUC+F1=1.0757)
...

═════════════════════════════════════════════════════════════════
FINAL TEST-SET RESULTS
═════════════════════════════════════════════════════════════════
  Optimal threshold : 0.3124
  AUC-ROC           : 0.7380
  Accuracy          : 0.7541
  Precision         : 0.5102
  Recall            : 0.6318
  F1-Score          : 0.5645
  MAE               : 0.2031
─────────────────────────────────────────────────────────────────
  Confusion Matrix  (rows=actual, cols=predicted):
    TP:   4,375  │  FP:   4,198
    FN:   2,544  │  TN:  26,520
═════════════════════════════════════════════════════════════════
```

---

## Evaluation Metrics Explained

| Metric | Description | Target |
|---|---|---|
| **AUC-ROC** | Ranking ability regardless of threshold | > 0.70 |
| **Accuracy** | Overall correct predictions | > 0.75 |
| **Precision** | Of predicted defaults, % that are actual defaults | > 0.50 |
| **Recall** | Of actual defaults, % that were caught | > 0.60 |
| **F1-Score** | Harmonic mean of precision and recall | > 0.55 |
| **MAE** | Mean absolute error of probability predictions | < 0.25 |

---

## Output Files

```
training/results/
├── best_tab_transformer.pth      # Best model weights (reloadable)
├── metrics.yaml                  # Final test-set metrics (machine-readable)
└── history_YYYYMMDD_HHMMSS.json  # Per-epoch loss, AUC, F1, recall curves
```

---

## Project Structure

```
training/
├── run_tab_transformer.py          # ← Entry point: run this
├── config.py                       # Python CONFIG dict (used by other models)
├── requirements.txt                # Base dependencies
│
├── src/
│   ├── load_data.py                # Data loading, filtering, splitting
│   ├── tab_transformer.py          # Model architecture (no changes needed)
│   ├── train_tab_transformer.py    # Preprocessing, training loop, evaluation
│   └── test_tab_transformer.py     # Unit tests for model components
│
├── config/
│   └── tab_transformer_config.yaml # All hyperparameters and feature lists
│
├── data/
│   └── accepted_2007_to_2018Q4.csv # Downloaded automatically by load_data.py
│
└── results/
    ├── best_tab_transformer.pth
    ├── metrics.yaml
    └── history_*.json
```

---

## Configuration Reference

All settings are in `config/tab_transformer_config.yaml`. Key sections:

```yaml
# Columns loaded from raw CSV (39 total)
data_settings:
  essential_columns: [loan_amnt, int_rate, term, emp_length, ...]
  years: [2013, 2014]

# Explicit feature type lists — drives the two model pathways
categorical_features: [term, emp_length, home_ownership, ...]  # 7 features
numerical_features:   [loan_amnt, int_rate, annual_inc, ...]   # 33 features

# Model architecture
model:
  embedding_dim: 32
  depth: 4
  heads: 4        # must always be <= len(categorical_features)
  dim_head: 32
  mlp_dim: 256
  dropout: 0.2

# Training
batch_size: 256
epochs: 100
learning_rate: 0.0005
weight_decay: 0.01
early_stopping_patience: 15
```

> **Note on `heads`:** This value must always be ≤ the number of categorical features (currently 7). The trainer clamps it automatically, but setting it higher than 7 in the config is harmless.

---

## Troubleshooting

**Low Recall (missing too many defaults)**
- Lower `early_stopping_patience` so the threshold-tuning runs more often
- The threshold is automatically optimised via the PR curve — check `metrics.yaml` to see what threshold was chosen

**Overfitting (loss falls, AUC plateaus)**
- Increase `dropout` (0.2 → 0.3)
- Increase `weight_decay` (0.01 → 0.05)
- Reduce `depth` (4 → 2) or `embedding_dim` (32 → 16)

**Training is slow**
- Ensure a GPU is available (`torch.cuda.is_available()`)
- Reduce `batch_size` if GPU memory is limited
- Reduce `depth` and `heads`

**Kaggle download fails**
- Ensure `~/.kaggle/kaggle.json` exists with your API token
- Or manually download from [kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place the CSV in `training/data/`

**`ModuleNotFoundError: No module named 'einops'`**
- Run: `pip install torch einops pyyaml`

---

## References

- [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)
- [Lending Club Dataset — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## License

MIT License
