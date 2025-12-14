# 🏗️ System Architecture

This document describes the **machine learning, software, and deployment architecture** of the Credit Risk Prediction System.

---

## 🔄 End-to-End Flow

```
┌────────────┐
│ User Input │
└─────┬──────┘
      │
      ▼
┌───────────────┐
│ Gradio UI     │  app.py
└─────┬─────────┘
      │
      ▼
┌────────────────────┐
│ Predictor Interface │  predictor.py
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Feature Validation  │
│ + Missing Handling  │
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Imputer → Scaler    │  (saved artifacts)
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ XGBoost Classifier  │
└─────┬──────────────┘
      │
      ▼
┌────────────────────┐
│ Probability Output  │
│ + Threshold Logic   │
└─────┬──────────────┘
      │
      ▼
┌───────────────┐
│ Decision      │  Approve / Reject
└───────────────┘
```

---

## 🧠 ML Architecture

### Model
- **Algorithm**: XGBoost (binary classifier)
- **Objective**: Predict probability of loan default
- **Metric**: AUC-ROC
- **Threshold**: 0.28 (profit-optimized)

### Feature Engineering

- Manual feature selection guided by:
  - LendingClub data dictionary
  - Financial interpretability
  - Leakage avoidance
- All transformations serialized and reused in inference

### Leakage Prevention

- No post-origination variables
- Time-based train/validation split
- Inference-time feature contract enforcement

---

## 🧩 Software Architecture

```
deployment/
├── app.py               # Gradio UI
├── predictor.py         # Inference logic
├── model_artifacts/     # Serialized pipeline
│   ├── imputer.pkl
│   ├── scaler.pkl
│   └── xgb_best_model.pkl
├── requirements.txt
└── Dockerfile
```

### predictor.py Responsibilities

- Load serialized artifacts
- Validate feature presence
- Apply preprocessing
- Produce calibrated probabilities
- Apply decision threshold

---

## 🚢 Deployment Architecture

```
GitHub Repo
   │
   │ push to main
   ▼
GitHub Actions
   │
   │ subtree split (deployment/)
   ▼
Hugging Face Space
   │
   ▼
Public Gradio App
```

---

## 🛡️ Design Principles

- **Reproducibility**
- **Separation of concerns**
- **Inference safety**
- **Business-aware ML**
- **Production realism**

---

## 📌 Future Extensions

- Model monitoring & drift detection
- SHAP-based explanations in UI
- Multi-model ensemble
- CI-based model re-training

---

## 📖 References

- LendingClub Loan Data
- XGBoost Documentation
- Quantitative Finance & Economics (2022)

