# 🏦 Credit Risk Prediction System (Training + Streamlit Deployment)

This project implements an **end-to-end credit risk prediction pipeline**:

* Train traditional ML models (LogReg / RF / XGBoost) on LendingClub data
* Export trained artifacts (model / scaler / imputer / feature contract)
* Run a **Streamlit UI with prediction + LLM assistant**

The workflow uses **two separate Python environments**:

* Python **3.14 environment → model training**
* Python **deployment environment → inference + UI**

---

## Project Structure

```
credit-risk-prediction-project/
│
├── training/
│   ├── train.py
│   ├── load_data.py
│   ├── build_features.py
│   ├── train_models.py
│   ├── find_features.py
│   ├── config.py
│   ├── requirements.txt
│   ├── models/          ← trained artifacts saved here
│   └── results/
│
├── deployment/
│   ├── streamlit_app.py
│   ├── predictor.py
│   ├── financial_sandbox.py
│   ├── requirements.txt
│   └── model_artifacts/ ← artifacts used for inference
│
└── README.md
```

---

# Step 1 — Training Environment (Python 3.14)

### 1. Create training virtual environment

```bash
cd credit-risk-prediction-project/training

python3.14 -m venv venv_train
source venv_train/bin/activate
```

---

### 2. Install training dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train models

Full training:

```bash
python train.py --full
```

Quick dev run:

```bash
python train.py --quick
```

This will:

* Download LendingClub dataset (via Kaggle CLI)
* Perform feature engineering
* Train multiple models
* Select best model (usually optimized XGBoost)
* Run business profit threshold analysis
* Save artifacts into:

```
training/models/
```

Artifacts include:

* `xgb_optimized_*.pkl`
* `scaler_*.pkl`
* `imputer_*.pkl`
* `model_comparison_*.csv`

---

### 4. Generate feature contract for deployment

After training finishes:

```bash
python find_features.py
```

This will create:

```
deployment/model_artifacts/training_features.json
deployment/model_artifacts/training_features.csv
```

---

# Step 2 — Deployment Environment (Streamlit UI)

Training uses Python 3.14,
but deployment should use a **stable Python version (e.g. 3.12)**.

---

### 1. Create deployment virtual environment

```bash
cd ../deployment

python3.12 -m venv venv_deploy
source venv_deploy/bin/activate
```

---

### 2. Install deployment dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Set OpenAI API key

This is required for the LLM chatbot panel.

Temporary (per terminal session):

```bash
export OPENAI_API_KEY="your_key_here"
```

Recommended (persistent):

```bash
nano ~/.zshrc
```

Add:

```
export OPENAI_API_KEY="your_key_here"
```

Then:

```bash
source ~/.zshrc
```

---

### 4. Run Streamlit app

```bash
streamlit run streamlit_app.py
```

Open browser:

```
http://localhost:8501
```

---

# Streamlit Interface

The UI contains two main panels:

### Left Panel — Credit Risk Form

* Input borrower financial + credit profile
* Model predicts default probability
* Uses business-optimized decision threshold

### Right Panel — LLM Chat Sandbox

* Powered by OpenAI / LangGraph agent logic
* Can answer finance questions
* Can trigger prediction tool
* Maintains session memory

---

# Artifact Sync Logic

When Streamlit starts:

* `predictor.py` automatically finds the **latest trained model**
* Copies artifacts from:

```
training/models/
```

→ into

```
deployment/model_artifacts/
```

This ensures deployment always uses **most recent training output**.

---

# Important Notes

* Training dataset is large (~1.2GB)
* Requires Kaggle CLI installed
* Model artifacts are version-sensitive (sklearn / xgboost warnings are expected)
* Do NOT mix training and deployment environments
* Do NOT commit API keys

---

# Minimal Run Summary

```bash
# TRAIN
cd training
python3.14 -m venv venv_train
source venv_train/bin/activate
pip install -r requirements.txt
python train.py --full
python find_features.py

# DEPLOY
cd ../deployment
python3.12 -m venv venv_deploy
source venv_deploy/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="xxx"
streamlit run streamlit_app.py
```
