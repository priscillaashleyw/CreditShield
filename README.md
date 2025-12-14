# Credit Risk Prediction System

### *Enhancing the 2022 paper: “Credit scoring for peer-to-peer lending using machine learning techniques”*

This project replicates and significantly improves the methodology from the 2022 Quantitative Finance & Economics paper on P2P loan credit scoring.
Using **358,244 LendingClub loans (2013–2014)**, the system predicts loan defaults with **92.3% AUC-ROC**, exceeding the paper’s 86–87% result.

The full ML pipeline includes data loading, preprocessing, feature engineering, model training, Bayesian optimization, business threshold optimization, and a deployable Gradio web app.

---

## Repository Structure

```
credit-risk-prediction/
│
├── training/
│   ├── train.py
│   ├── train_models.py
│   ├── load_data.py
│   ├── build_features.py
│   ├── find_features.py
│   ├── inspect_model.py
│   ├── config.py
│   └── requirements.txt
│
├── deployment/
│   ├── predictor.py
│   ├── gradio_app.py
│   ├── model_artifacts/
│   │     ├── model_xgb.pkl
│   │     ├── scaler.pkl
│   │     ├── imputer.pkl
│   │     └── training_features.json
│   └── requirements.txt
│
├── data/                         
├── models/                       
└── results/                      
```

---

## Installation

### 1. Install training environment

```bash
cd training
pip install -r requirements.txt
```

### 2. Install deployment environment

```bash
cd deployment
pip install -r requirements.txt
```

---

## Dataset

* LendingClub **accepted_2007_to_2018Q4.csv**
* Filtered to **2013–2014**
* Removed incomplete statuses
* → **358,244 final rows**

---

## Methodology

### Replicated from the original paper:

* Use of FICO, DTI, utilization, credit history variables
* Logistic Regression, XGBoost, Random Forest
* Feature selection via XGBoost gain
* No undersampling

### Improvements implemented:

* **Time-based split** instead of random
* **40+ engineered features**
* **NLP title features**
* **Interaction terms** + financial ratios
* **One-hot encoding** for purpose/state/home ownership
* **Bayesian Optimization (scikit-optimize)**
* **Business profit-max threshold (0.28)**
* **Deployable inference pipeline + Gradio app**

---

## Feature Engineering

### Paper’s 16 features:

Includes:
`dti`, `annual_inc`, `avg_cur_bal`, `total_bc_limit`,
`revol_util`, `revol_bal`, `fico_range_low`, `last_fico_range_high`,
`mths_since_recent_bc`, `mo_sin_old_rev_tl_op`, etc.

### Enhanced 40+ features:

| Category             | Examples                                             |
| -------------------- | ---------------------------------------------------- |
| Financial ratios     | `loan_to_income`, `int_rate_times_loan`              |
| Behavioral flags     | `has_delinq_history`, `subprime_high_dti`            |
| NLP features         | `title_has_debt`, `title_length`, `title_word_count` |
| Encoded categoricals | `addr_state_CA`, `purpose_debt_consolidation`        |
| Credit history time  | `years_since_earliest_cr`                            |

All training features are saved to:

```
deployment/model_artifacts/training_features.json
```

---

## Training the Model

In the `training/` folder:

```bash
python train.py --full
```

### Training modes:

| Mode            | Description                   |
| --------------- | ----------------------------- |
| `--quick`       | Fast training mode            |
| `--full`        | Full training + optimization  |
| `--sample N`    | Use N rows                    |
| `--no-optimize` | Disable Bayesian optimization |
| `--no-viz`      | Skip visualization            |

### Outputs saved:

* Trained model (`model_xgb.pkl`)
* `scaler.pkl`, `imputer.pkl`
* Feature list JSON
* ROC/PR curves
* Profit analysis CSV
* Model comparison tables

---

## Business Threshold Optimization

The model evaluates thresholds from 0.10 to 0.90 and computes:

* Profit from approving good borrowers
* Loss from approving bad borrowers
* Cost of rejecting good borrowers

The **optimal threshold = 0.28**, which is used in deployment.

---

## Running the Gradio App

In the `deployment/` folder:

```bash
python gradio_app.py
```

### App Features:

* User-friendly sliders + dropdowns
* Predicts APPROVE / REJECT
* Shows default probability
* Shows risk level + confidence
* Business explanation
* Colored risk bar visualization
* Example presets (low / high risk)

---

## Prediction API (programmatic use)

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
    "total_acc": 25
}

result = predictor.predict(loan)
print(result)
```

---

## 📈 Model Performance

| Metric      | Score                             |
| ----------- | --------------------------------- |
| **AUC-ROC** | **92.3%**                         |
| AUC-PR      | Strong performance with imbalance |
| Best model  | Optimized XGBoost                 |
| Validation  | Time-based split                  |

---

## Notes

* Raw LendingClub dataset is excluded (large file).
* Place trained artifacts into:

```
deployment/model_artifacts/
```

* For research & educational use only.

---

## License

Apache 2.0