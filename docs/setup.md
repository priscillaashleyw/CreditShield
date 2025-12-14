# ⚙️ Setup & Installation Guide

This guide explains how to **run training**, **launch the Gradio app**, and **deploy the system** locally or via Docker.

---

## 🧠 Prerequisites

- Python **3.10+** (3.11 recommended for deployment)
- Git & Git LFS
- (Optional) Docker

---

## 📦 Repository Setup

```bash
git clone <your-repo-url>
cd <repo-root>
```

Enable Git LFS:

```bash
git lfs install
git lfs pull
```

---

## 🏋️ Model Training

All training code lives in:

```
credit-risk-prediction-project/training/
```

### 1. Create Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training

Typical workflow:

- Data preprocessing
- Feature engineering
- Bayesian hyperparameter optimization (XGBoost)
- Threshold selection based on business utility

Artifacts produced:

```
model_artifacts/
├── xgb_best_model.pkl
├── imputer.pkl
├── scaler.pkl
├── training_features.csv
└── training_features.json
```

---

## 🌐 Running the Gradio App (Local)

```bash
cd credit-risk-prediction-project/deployment
pip install -r requirements.txt
python app.py
```

Open browser:

```
http://localhost:7860
```

---

## 🐳 Docker Deployment

```bash
docker build -t credit-risk-app .
docker run -p 7860:7860 credit-risk-app
```

---

## 🤗 Hugging Face Space Deployment

Deployment is **fully automated** via GitHub Actions.

### How it works:

- Any push to `main` affecting `deployment/`
- GitHub Actions triggers `.github/workflows/deploy.yml`
- `deployment/` subtree is pushed to Hugging Face Space

No manual steps required.

---

## 🧪 Programmatic Inference

```python
from predictor import CreditRiskPredictor

predictor = CreditRiskPredictor()
result = predictor.predict(input_dict)
```

---

## 🧯 Common Issues

- **Missing artifacts** → ensure `model_artifacts/` exists
- **Feature mismatch** → input keys must match training features
- **Docker build fails** → ensure `libgomp1` is installed

---

## 📌 Notes

- `.pkl` files are tracked via Git LFS
- Deployment artifacts are excluded from standard Git history
- Threshold tuned for **profit, not accuracy**

