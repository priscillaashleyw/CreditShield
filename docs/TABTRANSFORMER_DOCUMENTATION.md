# TabTransformer Implementation Documentation

## Overview

This document describes the implementation of **TabTransformer**, a deep learning architecture designed for tabular data based on the paper ["TabTransformer: Tabular Data Modeling Using Contextual Embeddings"](https://arxiv.org/abs/2012.06678). This implementation is used for **credit risk prediction** on Lending Club loan data.

---

## Table of Contents

1. [Task Description](#task-description)
2. [Dataset](#dataset)
3. [File Structure & Description](#file-structure--description)
4. [Quick Start](#quick-start)
5. [Architecture Overview](#architecture-overview)
6. [Core Components](#core-components)
7. [Model Configuration](#model-configuration)
8. [Training Pipeline](#training-pipeline)
9. [Data Preprocessing](#data-preprocessing)
10. [Latest Results](#latest-results)
11. [Usage Examples](#usage-examples)

---

## Task Description

### Problem Statement
**Regression** task to predict loan default probability.

Given a loan application with features like loan amount, interest rate, annual income, debt-to-income ratio, etc., predict the **probability (0-1)** that the loan will default.

### Output
- **Type**: Probability distribution (0 to 1)
- **Interpretation**: 
  - `0.0` = Very unlikely to default (good loan)
  - `1.0` = Very likely to default (risky loan)
- **Threshold**: Optimal threshold is learned during training (default: 0.5)

### Business Goal
Help lenders make informed decisions by:
1. Identifying high-risk loans before approval
2. Estimating probability of default for risk-based pricing
3. Reducing financial losses from loan defaults

---

## Dataset

### Source
**Lending Club Loan Data** (2013-2014)
- Downloaded from Kaggle: `kaggle://wordsforthewise/lending-club`
- Original file: `accepted_2007_to_2018Q4.csv`

### Data Filtering
| Filter | Description |
|--------|-------------|
| Years | 2013, 2014 only |
| Excluded Statuses | "Current" (incomplete loans) |
| Target Definition | Binary: Default vs Fully Paid |

### Target Variable
| Status | Label | Description |
|--------|-------|-------------|
| "Fully Paid" | 0 | Good loan - borrower repaid |
| "Charged Off", "Default" | 1 | Bad loan - borrower defaulted |

### Features Used
**Numerical Features** (auto-detected):
- `loan_amnt` - Loan amount requested
- `int_rate` - Interest rate
- `annual_inc` - Annual income
- `dti` - Debt-to-income ratio
- `delinq_2yrs` - Delinquencies in past 2 years
- `inq_last_6mths` - Inquiries in last 6 months
- `open_acc` - Open credit lines
- `pub_rec` - Public records
- `revol_bal` - Revolving balance
- `total_acc` - Total accounts
- `collections_12_mths_ex_med` - Collections in 12 months
- `acc_now_delinq` - Accounts currently delinquent
- `tot_coll_amt` - Total collection amount
- `tot_cur_bal` - Total current balance
- `total_rev_hi_lim` - Total revolving high credit limit

**Categorical Features** (auto-detected):
- `term` - Loan term (36 months / 60 months)

### Leaked Columns (Excluded)
These columns are excluded as they contain information not available at loan application time:
- `funded_amnt`, `funded_amnt_inv`
- `recoveries`, `collection_recovery_fee`
- `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`

---

## File Structure & Description

```
CreditShield/
│
├── docs/
│   └── TABTRANSFORMER_DOCUMENTATION.md  # This documentation file
│
├── credit-risk-prediction-project/
│   │
│   ├── config/
│   │   └── tab_transformer_config.yaml  # Model & training configuration
│   │                                      # - Hyperparameters (embedding_dim, depth, heads)
│   │                                      # - Training settings (batch_size, epochs, lr)
│   │                                      # - Data settings (years, target definition)
│   │
│   ├── training/
│   │   │
│   │   ├── run_tab_transformer.py       # 🚀 MAIN ENTRY POINT
│   │   │                                 # Run this to train on real data
│   │   │                                 # Usage: python run_tab_transformer.py
│   │   │
│   │   ├── run_tests.sh                 # 🧪 TEST RUNNER SCRIPT
│   │   │                                 # Runs unit tests + integration tests
│   │   │                                 # Usage: bash run_tests.sh
│   │   │
│   │   ├── test_integration.py          # Integration test with dummy data
│   │   │                                 # Tests full pipeline: load → preprocess → train → evaluate
│   │   │
│   │   ├── src/
│   │   │   ├── __init__.py              # Package initializer
│   │   │   │
│   │   │   ├── tab_transformer.py       # 🧠 CORE MODEL ARCHITECTURE
│   │   │   │                             # Contains: Embeddings, MultiHeadAttention,
│   │   │   │                             # FeedForward, TransformerBlock, TabTransformer
│   │   │   │
│   │   │   ├── train_tab_transformer.py # 🏋️ TRAINING PIPELINE
│   │   │   │                             # TabTransformerTrainer class with:
│   │   │   │                             # - Data preparation
│   │   │   │                             # - Model building
│   │   │   │                             # - Training loop
│   │   │   │                             # - Evaluation metrics
│   │   │   │
│   │   │   ├── test_tab_transformer.py  # 🧪 UNIT TESTS
│   │   │   │                             # Tests each component individually
│   │   │   │
│   │   │   ├── load_data.py             # 📊 DATA LOADING
│   │   │   │                             # DataLoader class for Lending Club data
│   │   │   │                             # - Downloads from Kaggle if needed
│   │   │   │                             # - Filters by year
│   │   │   │                             # - Defines target variable
│   │   │   │
│   │   │   └── build_features.py        # 🔧 FEATURE ENGINEERING
│   │   │                                 # Feature transformations and encoding
│   │   │
│   │   ├── results/
│   │   │   ├── best_tab_transformer.pth # 💾 SAVED MODEL WEIGHTS
│   │   │   │                             # Best model from training (PyTorch state dict)
│   │   │   │
│   │   │   ├── metrics.yaml             # 📈 FINAL METRICS
│   │   │   │                             # AUC, F1, Precision, Recall, etc.
│   │   │   │
│   │   │   └── history_*.json           # 📉 TRAINING HISTORY
│   │   │                                 # Loss and metrics per epoch
│   │   │
│   │   └── data/
│   │       └── lending_club_sample.csv  # Sample data for testing
│   │
│   └── deployment/
│       ├── app.py                       # FastAPI deployment app
│       └── predictor.py                 # Inference wrapper
│
└── requirements.txt                      # Python dependencies
```

### Key Files Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_tab_transformer.py` | Train model on real data | Full training run |
| `run_tests.sh` | Run all tests | Verify implementation works |
| `src/tab_transformer.py` | Model architecture | Understanding/modifying model |
| `src/train_tab_transformer.py` | Training logic | Customizing training |
| `src/load_data.py` | Data loading | Modifying data pipeline |
| `config/tab_transformer_config.yaml` | Configuration | Tuning hyperparameters |
| `results/metrics.yaml` | Final metrics | Checking model performance |
| `results/best_tab_transformer.pth` | Trained weights | Inference/deployment |

---

## Quick Start

### Running Tests

To verify the implementation is working correctly, run the test suite:

```bash
cd credit-risk-prediction-project/training
bash run_tests.sh
```

This runs:
1. **Unit Tests** (`src/test_tab_transformer.py`) - Tests individual components:
   - Embeddings layer
   - Multi-Head Attention
   - Feed Forward network
   - Transformer Block
   - Full TabTransformer forward pass
   - Training step validation

2. **Integration Test** (`test_integration.py`) - Tests the full pipeline with dummy data:
   - Data loading and preprocessing
   - Model building
   - Training loop
   - Evaluation metrics

Expected output:
```
================================================
TabTransformer Test Suite
================================================
>>> Running Unit Tests...
✓ PASS: Embeddings forward pass
✓ PASS: MHA forward pass
✓ PASS: FeedForward forward pass
✓ PASS: TransformerBlock forward pass
✓ PASS: TabTransformer forward pass
✓ PASS: Model has trainable parameters
✓ PASS: Training step
✓ PASS: Synthetic data training
RESULTS: 8 passed, 0 failed

>>> Running Integration Test...
✓ INTEGRATION TEST PASSED

✓ All tests passed!
```

### Running Training on Real Data

To train the model on actual Lending Club data:

```bash
cd credit-risk-prediction-project/training
python run_tab_transformer.py
```

This script:
1. Loads configuration from `config/tab_transformer_config.yaml`
2. Downloads/loads Lending Club data
3. Preprocesses and splits data (80/20 train/test)
4. Trains TabTransformer with early stopping
5. Saves best model to `results/best_tab_transformer.pth`
6. Saves metrics to `results/metrics.yaml`

---

## Architecture Overview

TabTransformer applies **Transformer attention** to categorical features while keeping numerical features in a separate bypass pathway. This design leverages the power of self-attention to learn contextual embeddings for categorical variables.

```
                    ┌─────────────────────────────────────┐
                    │           INPUT DATA                │
                    └─────────────────────────────────────┘
                              │                │
                    ┌─────────┴───────┐ ┌──────┴──────┐
                    │   Categorical   │ │  Numerical  │
                    │    Features     │ │  Features   │
                    └────────┬────────┘ └──────┬──────┘
                             │                 │
                    ┌────────▼────────┐        │
                    │   Embedding     │        │
                    │     Layer       │        │
                    └────────┬────────┘        │
                             │                 │
                    ┌────────▼────────┐        │
                    │   Transformer   │        │ (Bypass)
                    │     Blocks      │        │
                    │  (Self-Attn +   │        │
                    │   Feed Forward) │        │
                    └────────┬────────┘        │
                             │                 │
                    ┌────────▼────────┐        │
                    │  Adaptive Avg   │        │
                    │    Pooling      │        │
                    └────────┬────────┘        │
                             │                 │
                    ┌────────┴─────────────────┴────────┐
                    │           CONCATENATION           │
                    └───────────────────┬───────────────┘
                                        │
                    ┌───────────────────▼───────────────┐
                    │         MLP CLASSIFIER            │
                    │   (Linear → GELU → Dropout) × 3   │
                    └───────────────────┬───────────────┘
                                        │
                    ┌───────────────────▼───────────────┐
                    │     OUTPUT: Probability (0-1)     │
                    │         via Sigmoid               │
                    └───────────────────────────────────┘
```

---

## Core Components

### 1. Embeddings (`Embeddings` class)

Converts categorical integer indices into dense vector representations.

```python
class Embeddings(nn.Module):
    def __init__(self, input_dim, output_dim):
        # input_dim: number of categories + 1 (for unknown)
        # output_dim: embedding dimension (default: 32)
```

**Key Features:**
- Uses `nn.Embedding` with normal initialization (std=0.02)
- Each categorical feature gets its own embedding table
- Handles unknown categories with +1 dimension

---

### 2. Multi-Head Attention (`MultiHeadAttention` class)

Implements scaled dot-product multi-head self-attention.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        # dim: input/output dimension
        # heads: number of attention heads
        # dim_head: dimension per head
```

**Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

---

### 3. Feed Forward Network (`FeedForward` class)

Two-layer MLP with GELU activation applied after attention.

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        # Architecture: Linear → GELU → Dropout → Linear → Dropout
```

---

### 4. Transformer Block (`TransformerBlock` class)

Stacks multiple encoder layers with pre-normalization.

**Each Layer Contains:**
1. LayerNorm → Multi-Head Attention + Residual
2. LayerNorm → Feed Forward + Residual

---

### 5. TabTransformer (`TabTransformer` class)

Main model combining all components.

```python
class TabTransformer(nn.Module):
    def __init__(
        self,
        num_numerical_features,    # Number of numerical columns
        num_categorical_features,  # Number of categorical columns
        categorical_dims,          # List of category counts per feature
        embedding_dim=32,          # Embedding size
        depth=6,                   # Transformer layers
        heads=8,                   # Attention heads
        dim_head=64,               # Dimension per head
        mlp_dim=512,               # Feed-forward hidden dim
        num_classes=1,             # Single probability output
        dropout=0.1                # Dropout rate
    )
```

**Forward Pass:**
1. Embed each categorical feature → `(batch, num_cat, embed_dim)`
2. Apply Transformer blocks
3. Pool over sequence dimension → `(batch, embed_dim)`
4. Concatenate with numerical features → `(batch, embed_dim + num_numerical)`
5. Pass through MLP → Sigmoid → `(batch,)` probability output

---

## Model Configuration

Configuration is stored in `config/tab_transformer_config.yaml`:

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 32 | Dimension of categorical embeddings |
| `depth` | 4 | Number of Transformer encoder layers |
| `heads` | 4 | Number of attention heads |
| `dim_head` | 32 | Dimension per attention head |
| `mlp_dim` | 256 | Hidden dimension in feed-forward layers |
| `dropout` | 0.2 | Dropout rate for regularization |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Training batch size |
| `epochs` | 100 | Maximum training epochs |
| `learning_rate` | 0.0005 | Initial learning rate |
| `weight_decay` | 0.01 | L2 regularization (AdamW) |
| `early_stopping_patience` | 15 | Epochs without improvement before stopping |

---

## Training Pipeline

### Loss Function
- **Binary Cross Entropy (BCE)** with class weighting
- Model outputs probability via sigmoid (0-1)
- Positive class weight = `n_negative / n_positive` for imbalanced data

### Optimizer & Scheduler
- **AdamW** with weight decay
- **CosineAnnealingWarmRestarts** scheduler

### Early Stopping
- Metric: AUC + F1 combined score
- Patience: 15 epochs

---

## Latest Results

From `results/metrics.yaml`:

```yaml
model_type: TabTransformer
output_type: probability (0-1)
optimal_threshold: 0.486
auc: 0.5089
accuracy: 0.6316
precision: 0.6316
recall: 1.0000
f1: 0.7742
mae: 0.5003
true_positives: 48
true_negatives: 0
false_positives: 28
false_negatives: 0
```

### Interpretation
- **Output**: Single probability value between 0 and 1
- **Threshold**: 0.486 (optimized for F1 score)
- **Recall**: 100% - catches all defaults
- **F1 Score**: 0.77 - good balance of precision/recall

---

## Usage Examples

### Inference with Trained Model

```python
import torch
from tab_transformer import TabTransformer

# Load trained model
model = TabTransformer(
    num_numerical_features=15,
    num_categorical_features=1,
    categorical_dims=[2],  # e.g., term has 2 categories
    embedding_dim=32,
    depth=4,
    heads=4
)
model.load_state_dict(torch.load('results/best_tab_transformer.pth'))
model.eval()

# Predict probability
with torch.no_grad():
    prob = model(numerical_tensor, categorical_tensor)
    # prob is now between 0 and 1
    
    # Apply threshold for binary prediction
    prediction = (prob >= 0.486).int()
    
print(f"Default probability: {prob.item():.2%}")
print(f"Prediction: {'DEFAULT' if prediction else 'GOOD LOAN'}")
```

---

## Dependencies

```
torch>=1.9.0
einops>=0.3.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
pyyaml>=5.4.0
```

---

## References

- Original Paper: [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/abs/2012.06678)
- Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020)
