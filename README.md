# 🏦 Credit Risk Prediction System (End-to-End ML + Deployment)

> **A production-ready credit risk prediction system** that replicates and significantly improves a 2022 academic paper on P2P lending credit scoring, with a full training pipeline, business-aware threshold optimization, and an auto-deployed Gradio web app on Hugging Face Spaces.

---

## 🔍 Project Overview

This repository contains an **end-to-end machine learning system for predicting loan default risk** using LendingClub data. The project:

- Replicates the 2022 paper:  
  *“Machine learning and artificial neural networks to construct P2P lending credit-scoring model: A case using Lending Club data”*  
  (*Quantitative Finance & Economics*)
- **Improves AUC-ROC from ~86–87% to 92.3%**
- Uses **time-based validation**, extensive feature engineering, and **profit-optimized decision thresholds**
- Provides a **fully deployable Gradio application**, automatically synced to Hugging Face via GitHub Actions

This repo is designed to be useful for:
- ML / data science portfolios
- Research replication & extension
- Demonstrating production ML (training → artifacts → inference → deployment)

---

## 🧠 Key Highlights

- **92.3% AUC-ROC** on 358,244 real loans
- **No undersampling** (handles real-world class imbalance)
- **40+ engineered features** (ratios, flags, NLP title features)
- **Bayesian hyperparameter optimization**
- **Business-aware threshold optimization** (profit-maximizing, not accuracy-maximizing)
- **Single-source-of-truth feature list** shared between training & inference
- **Auto-deployed Gradio app** via GitHub → Hugging Face Spaces
- **Dockerized deployment** for reproducibility

---

## 📁 Repository Structure

```
.
├── .github/workflows/
│   └── deploy.yml                  # CI: sync deployment/ to Hugging Face
│
├── credit-risk-prediction-project/
│   ├── training/                   # Model training & experimentation
│   │   ├── train.py
│   │   ├── train_models.py
│   │   ├── load_data.py
│   │   ├── build_features.py
│   │   ├── find_features.py
│   │   ├── inspect_model.py
│   │   ├── config.py
│   │   └── requirements.txt
│   │
│   └── deployment/                 # Production inference & UI
│       ├── app.py                  # Gradio app (same as gradio_app.py)
│       ├── predictor.py            # Feature engineering + inference pipeline
│       ├── requirements.txt
│       ├── Dockerfile
│       ├── README.md               # Hugging Face Space metadata
│       └── model_artifacts/
│           ├── xgb_best_model.pkl
│           ├── scaler.pkl
│           ├── imputer.pkl
│           ├── training_features.json
│           └── training_features.csv
│
├── LCDataDictionary.xlsx            # LendingClub official feature dictionary
├── .gitattributes                  # Git LFS config for model files
├── .gitignore
└── README.md                       # ← you are here
```

---

## 📊 Dataset

- **Source**: LendingClub public loan dataset
- **File**: `accepted_2007_to_2018Q4.csv` (not included)
- **Filtered period**: **2013–2014**
- **Final sample size**: **358,244 loans**
- **Target**: Loan default / charged-off status

> 📘 `LCDataDictionary.xlsx` provides the official meaning of all LendingClub variables used in training and feature engineering.

---

## 🧪 Methodology

### Replicated from the Paper

- Core credit variables (DTI, FICO, utilization, balances)
- Logistic Regression, Random Forest, XGBoost baselines
- Feature importance–based selection
- No class rebalancing

### Improvements in This Project

| Area | Improvement |
|----|----|
| Validation | **Time-based split** (no leakage) |
| Features | 40+ engineered features |
| Categoricals | One-hot encoded state, purpose, home ownership |
| NLP | Loan title keyword + length features |
| Optimization | Bayesian hyperparameter tuning |
| Decisioning | **Profit-maximizing threshold (0.28)** |
| Deployment | Shared feature contract + Gradio UI |

---

## 🛠 Feature Engineering (Overview)

### Examples of Engineered Features

| Category | Examples |
|-------|---------|
| Financial ratios | `loan_to_income`, `int_rate_times_loan` |
| Credit behavior | `has_delinq_history`, `subprime_high_dti` |
| Credit history | `years_since_earliest_cr` |
| NLP (title) | `title_has_debt`, `title_word_count` |
| One-hot | `addr_state_CA`, `purpose_debt_consolidation` |

🔑 **Important**:  
All features used during training are saved to:

```
credit-risk-prediction-project/deployment/model_artifacts/training_features.json
```

This file is treated as the **single source of truth** for inference.

---

## 🚂 Training the Model

### Environment Setup

```bash
cd credit-risk-prediction-project/training
pip install -r requirements.txt
```

### Run Training

```bash
python train.py --full
```

#### Training Modes

| Flag | Description |
|----|------------|
| `--quick` | Fast dev run |
| `--full` | Full training + optimization |
| `--sample N` | Train on N rows |
| `--no-optimize` | Skip Bayesian tuning |
| `--no-viz` | Skip plots |

### Outputs

- Trained XGBoost model
- Scaler & imputer
- Feature list JSON/CSV
- ROC & PR curves
- Profit-by-threshold analysis

Artifacts intended for deployment should be copied to:

```
credit-risk-prediction-project/deployment/model_artifacts/
```

---

## 💰 Business Threshold Optimization

Instead of using a default 0.5 cutoff, the model:

- Simulates **profit vs loss** across thresholds 0.10–0.90
- Accounts for:
  - Approving good borrowers
  - Approving bad borrowers (loss)
  - Rejecting good borrowers (opportunity cost)

✅ **Optimal threshold**: **0.28**  
This threshold is **hard-coded into the predictor and UI**.

---

## 🚀 Running the Gradio App (Local)

### Option 1: Python

```bash
cd credit-risk-prediction-project/deployment
pip install -r requirements.txt
python app.py
```

Open: http://localhost:7860

### Option 2: Docker

```bash
docker build -t credit-risk-app .
docker run -p 7860:7860 credit-risk-app
```

---

## 🌐 Hugging Face Deployment (Auto)

This repo includes a **GitHub Actions workflow** that:

- Watches for changes in `deployment/`
- Uses `git subtree` to push **only deployment/**
- Syncs automatically to a Hugging Face Space

📁 Workflow file:
```
.github/workflows/deploy.yml
```

This ensures:
- Clean separation of training vs deployment
- No accidental leakage of training code or raw data

---

## 🧩 Predictor Architecture (Important)

`predictor.py` implements a **production-safe inference pipeline**:

1. Accepts raw user input
2. Infers missing fields with safe defaults
3. Recreates **all engineered & one-hot features**
4. Orders features to exactly match training
5. Applies imputer → scaler → model

This avoids common deployment failures due to feature mismatch.

---

## 🧪 Programmatic Usage

```python
from predictor import CreditRiskPredictor

predictor = CreditRiskPredictor("model_artifacts")

loan = {
    "loan_amnt": 15000,
    "int_rate": 12.5,
    "grade": "C",
    "emp_length": "5 years",
    "annual_inc": 75000,
    "dti": 18.5,
    "revol_util": "45%",
    "delinq_2yrs": 0,
    "inq_last_6mths": 2,
    "open_acc": 8,
    "total_acc": 25,
    "addr_state": "CA",
    "purpose": "debt_consolidation",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "title": "Debt consolidation loan"
}

result = predictor.predict(loan)
print(result)
```

---

## 📈 Model Performance

| Metric | Value |
|----|----|
| AUC-ROC | **92.3%** |
| Validation | Time-based |
| Model | Optimized XGBoost |
| Threshold | 0.28 (profit-max) |

---

## ⚠️ Notes & Limitations

- Raw LendingClub data is **not included**
- Model is trained on **historical data (2013–2014)**
- For **research & educational use only**
- Not financial or lending advice

---

## 📜 License

Apache 2.0

---

## ✨ Author & Purpose

This project demonstrates:
- Research replication done *properly*
- Strong ML engineering practices
- Realistic deployment constraints

If you’re reviewing this as a recruiter, researcher, or engineer:  
**everything from feature leakage prevention to inference robustness is intentional.**
