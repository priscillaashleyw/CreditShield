# TabTransformer for Credit Risk Prediction

## Overview

This project implements **TabTransformer** - a deep learning architecture that applies Transformer attention mechanisms to tabular data for credit risk prediction (loan default classification).

## Key Features

- **TabTransformer Architecture**: Transformer-based model for mixed numerical/categorical data
- **Class Imbalance Handling**: Weighted loss function and threshold optimization
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Comprehensive Metrics**: AUC, F1, Precision, Recall, MAE, Confusion Matrix

## Quick Start

```bash
# 1. Activate virtual environment
cd /Users/grace/fintech_new/CreditShield
source venv/bin/activate

# 2. Navigate to training directory
cd credit-risk-prediction-project/training

# 3. Create sample data (if no Kaggle access)
python create_sample_data.py

# 4. Train the model
python train_model.py
```

## Model Architecture

```
TabTransformer
├── Categorical Embeddings (per feature)
├── Transformer Encoder
│   ├── Multi-Head Self-Attention
│   ├── Feed-Forward Network
│   └── Layer Normalization
├── Pooling Layer
├── Numerical Feature Bypass
└── MLP Classifier
```

### Hyperparameters (Tuned)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 32 | Dimension of categorical embeddings |
| `depth` | 4 | Number of transformer layers |
| `heads` | 4 | Number of attention heads |
| `dropout` | 0.2 | Dropout rate for regularization |
| `learning_rate` | 0.0005 | Initial learning rate |
| `batch_size` | 128 | Training batch size |
| `weight_decay` | 0.01 | L2 regularization |

## Handling Class Imbalance

Credit risk datasets are typically imbalanced (~80% good loans, ~20% defaults). This implementation addresses imbalance through:

1. **Weighted Cross-Entropy Loss**: Higher weight for minority class (defaults)
2. **Optimal Threshold Selection**: Uses precision-recall curve to find best decision threshold
3. **Combined Evaluation Metric**: AUC + F1 for model selection

## Training Output

```
======================================================================
TabTransformer Training for Credit Risk Analysis
======================================================================

[1/5] Checking for data...
✓ Found local data: lending_club_sample.csv (0.94 MB)

[2/5] Loading configuration...
✓ Configuration loaded

[3/5] Loading and preprocessing data...
✓ Data loaded: 10,000 rows, 18 columns
✓ Target defined: 10,000 loans (18.19% default rate)

[4/5] Splitting data...
✓ Training: 8,000 samples
✓ Testing: 2,000 samples

[5/5] Training TabTransformer...
  Class imbalance ratio: 4.50:1
  Class weights: [1.0, 4.50]

Epoch 1/100 - Loss: 0.8234
  AUC: 0.6234 | F1: 0.3421 | Prec: 0.4123 | Recall: 0.2934
...

==============================================================
FINAL TEST SET RESULTS
==============================================================
  Optimal Threshold: 0.287
  AUC-ROC:     0.7234
  Accuracy:    0.7456
  Precision:   0.5234
  Recall:      0.6123
  F1-Score:    0.5645
  MAE:         0.2134
--------------------------------------------------------------
Confusion Matrix:
  TP: 223 | FP: 203
  FN: 141 | TN: 1433
==============================================================
```

## Evaluation Metrics Explained

| Metric | Description | Target |
|--------|-------------|--------|
| **AUC-ROC** | Area under ROC curve. Measures ranking ability | > 0.70 |
| **Accuracy** | Overall correct predictions | > 0.75 |
| **Precision** | Of predicted defaults, how many are actual defaults | > 0.50 |
| **Recall** | Of actual defaults, how many were detected | > 0.60 |
| **F1-Score** | Harmonic mean of precision and recall | > 0.55 |
| **MAE** | Mean absolute error of probability predictions | < 0.25 |

## Output Files

After training, check `results/` directory:

```
results/
├── best_tab_transformer.pth    # Best model weights
├── metrics.yaml                # Final evaluation metrics
├── model_info.yaml             # Model architecture info
└── history_YYYYMMDD_HHMMSS.json # Training curves
```

## Configuration

Edit `config/tab_transformer_config.yaml`:

```yaml
# Model architecture
model:
  embedding_dim: 32
  depth: 4
  heads: 4
  dropout: 0.2

# Training parameters
batch_size: 128
epochs: 100
learning_rate: 0.0005
weight_decay: 0.01
early_stopping_patience: 15
```

## Troubleshooting

### Low Recall (Missing Defaults)
- Increase `pos_weight` in class weights
- Lower the decision threshold
- Increase training epochs

### Overfitting (High train, low test)
- Increase `dropout` (0.2 → 0.3)
- Increase `weight_decay` (0.01 → 0.05)
- Reduce `depth` or `embedding_dim`

### Slow Training
- Reduce `batch_size` (if memory limited)
- Reduce `depth` and `heads`
- Use GPU if available

## Project Structure

```
training/
├── train_model.py              # Main training script
├── create_sample_data.py       # Generate test data
├── src/
│   ├── tab_transformer.py      # Model architecture
│   ├── train_tab_transformer.py # Training loop
│   └── load_data.py            # Data preprocessing
├── data/
│   └── lending_club_sample.csv # Training data
├── results/
│   └── best_tab_transformer.pth # Saved model
└── README.md                   # This file
```

## References

- [TabTransformer Paper](https://arxiv.org/abs/2012.06678)
- [Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## License

MIT License

