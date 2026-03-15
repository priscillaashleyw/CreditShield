# CreditShield Financial Sandbox — Tutorial

The Financial Sandbox is a conversational what-if simulator. You describe loan scenarios in plain English, and the agent runs them against a real XGBoost credit risk model trained on 358,000 Lending Club loans. The model returns a default probability, risk level, and APPROVE/REJECT decision.

> **Model note:** Trained on 2013–2014 Lending Club data with loan amounts up to $40,000. The approval threshold is 85% — only profiles with a default probability above 85% are rejected.

---

## Setup

```bash
export OPENAI_API_KEY='sk-or-v1-...'
python financial_sandbox.py
```

The model loads from `credit-risk-prediction-project/deployment/model_artifacts/` — no Gradio server or Docker required.

---

## Baseline Profile

When you start a session, the agent loads this default profile:

| Feature | Value |
|---------|-------|
| Loan amount | $15,000 |
| Interest rate | 12.5% |
| Grade | C |
| Annual income | $75,000 |
| DTI | 18.5% |
| FICO score | 680–684 |
| Home ownership | RENT |
| Purpose | Debt consolidation |

---

## Example Session

### 1. Check your current risk

```
User: What is my current credit risk?

Agent: Your current credit risk assessment indicates the following:

  Decision            APPROVE
  Default Probability 14.28%
  Risk Level          LOW
  Approval Threshold  85.00%

This means you have a low risk of default based on your current profile.
```

The baseline profile (grade C, $75k income, FICO 680) sits well below the 85% threshold.

---

### 2. What-if: improve your grade and income

```
User: If I upgrade to grade A and raise my income to $120k with DTI of 8, what happens?

Agent: If you upgrade to grade A while maintaining an income of $120,000 and a DTI of 8,
       the assessment would be:

  Decision            APPROVE
  Default Probability 5.93%
  Risk Level          LOW
  Approval Threshold  85.00%

  Current default probability:  14.28%
  New default probability:       5.93%
  Change:                       -8.35pp
```

The agent applies the changes to your profile, runs a new prediction, and compares against the previous result. The profile is updated for the rest of the session.

---

### 3. Layer on another change (FICO score)

The agent remembers the previous updates. You can keep layering changes:

```
User: What if I have a FICO score of 750 instead?

Agent: If you have a FICO score of 750, along with grade A, an income of $120,000,
       and a DTI of 8:

  Decision            APPROVE
  Default Probability 1.46%
  Risk Level          LOW
  Approval Threshold  85.00%

  Previous probability (without FICO update): 5.93%
  New probability (with FICO 750):            1.46%
  Change:                                    -4.47pp
```

FICO score is the single most influential feature in this model — improving it from 680 to 750 cuts the default probability by two-thirds on top of the grade/income improvements.

---

### 4. Reset and try a high-risk scenario

```
User: Reset to defaults, then what would happen if I had grade F, income $30k, and DTI 38?

Agent: Your profile has been reset. With grade F, income of $30,000, and DTI of 38:

  Decision            APPROVE
  Default Probability 32.17%
  Risk Level          MEDIUM
  Approval Threshold  85.00%
```

Even this poor profile is approved because the threshold is calibrated for profit maximization, not risk minimization — the model approves all loans it expects to be net-positive, and only rejects when default probability exceeds 85%. A grade F, low-income borrower with high DTI sits at MEDIUM risk but is still below that bar.

---

### 5. Clearly out-of-range input

```
User: I want a loan of $5 million but I only earn $30k a year. Will I get approved?

Agent: A loan of $5 million with an annual income of $30,000 is far outside the typical
       range for this model. The model was trained on loan amounts up to $40,000, and
       predictions for inputs this extreme are unreliable.

       Would you like me to run it anyway?
```

The agent flags out-of-distribution inputs rather than silently returning a meaningless result.

---

## Feature Reference

You can refer to any of these features in natural language:

| Feature | Valid values |
|---------|-------------|
| `grade` | `A` `B` `C` `D` `E` `F` `G` (A = best) |
| `home_ownership` | `RENT` `MORTGAGE` `OWN` `OTHER` |
| `purpose` | `debt_consolidation` `credit_card` `home_improvement` `major_purchase` `medical` `car` `wedding` |
| `verification_status` | `Verified` `Source Verified` `Not Verified` |
| `emp_length` | `< 1 year` `1 year` … `10+ years` |
| `addr_state` | 2-letter US state code (e.g. `CA` `NY` `TX`) |
| `loan_amnt` | Number, up to ~$40,000 (model training range) |
| `annual_inc` | Number |
| `int_rate` | Percentage as number (e.g. `12.5`) |
| `dti` | Debt-to-income ratio as number |
| `last_fico_range_low` / `last_fico_range_high` | 300–850 |

---

## Understanding the Results

| Field | Meaning |
|-------|---------|
| **Default Probability** | Model's estimated chance of loan default (0–100%) |
| **Risk Level** | LOW < 20% · MEDIUM 20–40% · HIGH 40–60% · VERY HIGH > 60% |
| **Threshold** | 85% — loans are rejected only above this |
| **Decision** | APPROVE if default probability < 85%, REJECT otherwise |

The threshold was selected from a profit-maximization analysis on the training data. It is intentionally permissive — the model approves borderline loans because the expected profit from good loans outweighs the loss from occasional defaults at that operating point.
