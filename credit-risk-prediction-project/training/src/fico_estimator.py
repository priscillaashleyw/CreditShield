"""
fico_estimator.py — Estimate last_fico_range_low from other credit profile features.

Uses Ridge regression trained on Lending Club data.
Output is clamped to the valid FICO range [300, 850].
last_fico_range_high is derived as estimated_low + 4 (standard LC gap).
"""

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features used to estimate FICO — all derivable from raw user inputs
# Mapped to FICO's 5 components:
#   Payment History (35%): pct_tl_nvr_dlq, delinq_2yrs
#   Credit Utilisation (30%): revol_util, revol_bal, total_bc_limit
#   Length of History (15%): years_since_earliest_cr, mo_sin_old_rev_tl_op, mo_sin_old_il_acct
#   Credit Mix (10%): total_acc
#   New Credit (10%): mths_since_recent_inq, mo_sin_rcnt_rev_tl_op
FICO_ESTIMATOR_FEATURES = [
    'revol_util',             # credit card utilisation %
    'pct_tl_nvr_dlq',         # % accounts never late  (payment history)
    'years_since_earliest_cr',# length of credit history
    'total_acc',              # total credit accounts  (credit mix proxy)
    'mths_since_recent_inq',  # months since last credit check (new credit)
    'mo_sin_old_rev_tl_op',   # months since oldest credit card
    'mo_sin_old_il_acct',     # months since oldest loan
    'revol_bal',              # revolving balance
    'total_bc_limit',         # total credit card limit
    'avg_cur_bal',            # average account balance
    'total_bal_ex_mort',      # total debt excl. mortgage
    'mths_since_recent_bc',   # months since newest bank card
    'mo_sin_rcnt_rev_tl_op',  # months since newest credit card
    'dti',                    # debt-to-income ratio
    'annual_inc',             # annual income
    'emp_length_numeric',     # employment length in years
    'delinq_2yrs',            # late payments in last 2 years
]

FICO_LOW_GAP = 4  # last_fico_range_high is always last_fico_range_low + 4 in LC data


def train_fico_estimator(df, save_path=None):
    """
    Train a Ridge regression to estimate last_fico_range_low.

    Args:
        df:         DataFrame with all features (post-engineering, includes target col)
        save_path:  Path to save the estimator pkl (optional)

    Returns:
        estimator bundle dict
    """
    target_col = 'last_fico_range_low'

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in dataframe")

    mask = df[target_col].notna()
    df_clean = df[mask].copy()
    print(f"   Training FICO estimator on {len(df_clean):,} samples")

    available_features = [f for f in FICO_ESTIMATOR_FEATURES if f in df_clean.columns]
    missing = set(FICO_ESTIMATOR_FEATURES) - set(available_features)
    if missing:
        print(f"   ⚠️  Missing features (will be skipped): {sorted(missing)}")
    print(f"   Using {len(available_features)} features")

    X = df_clean[available_features]
    y = df_clean[target_col]

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0)),
    ])
    pipeline.fit(X, y)

    y_pred = np.clip(pipeline.predict(X), 300, 850)
    mae = float(np.mean(np.abs(y_pred - y)))
    r2  = float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
    print(f"   Training MAE : {mae:.1f} FICO points")
    print(f"   Training R²  : {r2:.4f}")

    bundle = {
        'pipeline': pipeline,
        'features': available_features,
        'fico_low_gap': FICO_LOW_GAP,
        'mae': mae,
        'r2': r2,
    }

    if save_path:
        joblib.dump(bundle, save_path)
        print(f"   ✓ FICO estimator saved: {save_path}")

    return bundle


def _emp_length_to_numeric(emp_length_val):
    """Convert emp_length string to numeric years (mirrors predictor logic)."""
    if pd.isna(emp_length_val):
        return 3.0
    s = str(emp_length_val).lower()
    if '10+' in s:
        return 10.0
    if '< 1' in s:
        return 0.5
    nums = re.findall(r'\d+', s)
    return float(nums[0]) if nums else 3.0


def estimate_fico(bundle, input_dict):
    """
    Estimate last_fico_range_low and last_fico_range_high from raw user inputs.

    Args:
        bundle:     estimator bundle loaded from pkl
        input_dict: raw user input dict (same keys as the Streamlit form)

    Returns:
        (estimated_low: int, estimated_high: int)
    """
    pipeline  = bundle['pipeline']
    features  = bundle['features']
    gap       = bundle.get('fico_low_gap', FICO_LOW_GAP)

    # Build feature row — NaN for anything missing (imputer uses training medians)
    row = {f: input_dict.get(f, np.nan) for f in features}

    # Derive emp_length_numeric if the user provided emp_length string instead
    if 'emp_length_numeric' in features and np.isnan(row.get('emp_length_numeric', np.nan)):
        row['emp_length_numeric'] = _emp_length_to_numeric(input_dict.get('emp_length', ''))

    X = pd.DataFrame([row])[features]

    estimated_low  = float(np.clip(pipeline.predict(X)[0], 300, 850))
    estimated_high = float(min(850, estimated_low + gap))

    return int(round(estimated_low)), int(round(estimated_high))
