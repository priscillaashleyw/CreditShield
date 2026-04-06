#!/usr/bin/env python3
"""
demo_flex_transformer.py
========================
Runnable demo of FlexTabTransformer with real sample loan inputs.

Shows all four output modes:
  1. Binary classification  — P(default)
  2. Multi-class            — risk grade (A/B/C/D/E)
  3. Regression             — predicted loss amount
  4. Multi-output           — all three at once

Run from the training/ directory:
    python demo_flex_transformer.py
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.tab_transformer_flex import FlexTabTransformer


# =====================================================================
# Sample loan applications (3 borrowers)
# =====================================================================
# These mirror the exact features your pipeline uses.

SAMPLE_LOANS = [
    {
        # Borrower 1: Strong profile — likely GOOD loan
        "name":                "Alice (strong profile)",
        "term":                "36 months",
        "emp_length":          "10+ years",
        "home_ownership":      "MORTGAGE",
        "verification_status": "Verified",
        "purpose":             "debt_consolidation",
        "addr_state":          "CA",
        "initial_list_status": "w",
        # Numerical features
        "loan_amnt":            10000.0,
        "int_rate":             7.5,
        "annual_inc":           np.log1p(95000),   # log1p as load_data does
        "dti":                  12.5,
        "delinq_2yrs":         0,
        "inq_last_6mths":      0,
        "open_acc":            10,
        "pub_rec":             0,
        "revol_bal":           np.log1p(8500),
        "revol_util":          25.0,
        "total_acc":           22,
        "last_fico_range_high": 750,
        "last_fico_range_low":  745,
        "collections_12_mths_ex_med": 0,
        "acc_now_delinq":      0,
        "tot_coll_amt":        0,
        "tot_cur_bal":         45000,
        "total_rev_hi_lim":    35000,
        "avg_cur_bal":         12000,
        "bc_util":             30.0,
        "mort_acc":            1,
        "pub_rec_bankruptcies": 0,
        "tax_liens":           0,
        "tot_hi_cred_lim":     120000,
        "total_bal_ex_mort":   25000,
        "total_bc_limit":      20000,
        "total_il_high_credit_limit": 50000,
        "mths_since_last_delinq": 999,    # 999 = never happened
        "mths_since_last_record": 999,
        "mths_since_last_major_derog": 999,
        "mths_since_last_delinq_missing": 1,
        "mths_since_last_record_missing": 1,
        "mths_since_last_major_derog_missing": 1,
    },
    {
        # Borrower 2: Moderate risk
        "name":                "Bob (moderate risk)",
        "term":                "36 months",
        "emp_length":          "3 years",
        "home_ownership":      "RENT",
        "verification_status": "Not Verified",
        "purpose":             "credit_card",
        "addr_state":          "TX",
        "initial_list_status": "f",
        "loan_amnt":            20000.0,
        "int_rate":             15.0,
        "annual_inc":           np.log1p(55000),
        "dti":                  22.0,
        "delinq_2yrs":         1,
        "inq_last_6mths":      3,
        "open_acc":            8,
        "pub_rec":             0,
        "revol_bal":           np.log1p(18000),
        "revol_util":          72.0,
        "total_acc":           15,
        "last_fico_range_high": 680,
        "last_fico_range_low":  675,
        "collections_12_mths_ex_med": 0,
        "acc_now_delinq":      0,
        "tot_coll_amt":        0,
        "tot_cur_bal":         22000,
        "total_rev_hi_lim":    25000,
        "avg_cur_bal":         5500,
        "bc_util":             68.0,
        "mort_acc":            0,
        "pub_rec_bankruptcies": 0,
        "tax_liens":           0,
        "tot_hi_cred_lim":     45000,
        "total_bal_ex_mort":   22000,
        "total_bc_limit":      12000,
        "total_il_high_credit_limit": 20000,
        "mths_since_last_delinq": 18,     # had a delinquency 18 months ago
        "mths_since_last_record": 999,
        "mths_since_last_major_derog": 999,
        "mths_since_last_delinq_missing": 0,
        "mths_since_last_record_missing": 1,
        "mths_since_last_major_derog_missing": 1,
    },
    {
        # Borrower 3: High risk — likely to DEFAULT
        "name":                "Charlie (high risk)",
        "term":                "60 months",
        "emp_length":          "< 1 year",
        "home_ownership":      "RENT",
        "verification_status": "Not Verified",
        "purpose":             "small_business",
        "addr_state":          "FL",
        "initial_list_status": "f",
        "loan_amnt":            35000.0,
        "int_rate":             24.0,
        "annual_inc":           np.log1p(32000),
        "dti":                  28.5,
        "delinq_2yrs":         3,
        "inq_last_6mths":      5,
        "open_acc":            12,
        "pub_rec":             1,
        "revol_bal":           np.log1p(29000),
        "revol_util":          95.0,
        "total_acc":           18,
        "last_fico_range_high": 620,
        "last_fico_range_low":  615,
        "collections_12_mths_ex_med": 1,
        "acc_now_delinq":      1,
        "tot_coll_amt":        1500,
        "tot_cur_bal":         35000,
        "total_rev_hi_lim":    30000,
        "avg_cur_bal":         4200,
        "bc_util":             92.0,
        "mort_acc":            0,
        "pub_rec_bankruptcies": 1,
        "tax_liens":           0,
        "tot_hi_cred_lim":     55000,
        "total_bal_ex_mort":   35000,
        "total_bc_limit":      8000,
        "total_il_high_credit_limit": 15000,
        "mths_since_last_delinq": 4,      # recent delinquency
        "mths_since_last_record": 24,
        "mths_since_last_major_derog": 12,
        "mths_since_last_delinq_missing": 0,
        "mths_since_last_record_missing": 0,
        "mths_since_last_major_derog_missing": 0,
    },
]


# =====================================================================
# Feature definitions (must match tab_transformer_config.yaml)
# =====================================================================

CATEGORICAL_COLS = [
    "term", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state",
    "initial_list_status",
]

NUMERICAL_COLS = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "last_fico_range_high", "last_fico_range_low",
    "collections_12_mths_ex_med", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "avg_cur_bal", "bc_util", "mort_acc",
    "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit",
    "mths_since_last_delinq", "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_last_delinq_missing",
    "mths_since_last_record_missing",
    "mths_since_last_major_derog_missing",
]

RISK_GRADES = ["A", "B", "C", "D", "E"]


def prepare_inputs(loans):
    """
    Convert raw loan dicts into tensors ready for the model.

    This mimics what TabTransformerTrainer.prepare_data() does:
      - LabelEncode categoricals
      - StandardScale numericals
    """
    # ── Categorical: label-encode ─────────────────────────────────────
    label_encoders = {}
    cat_encoded = []

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        vals = [str(loan[col]) for loan in loans]
        le.fit(vals)
        cat_encoded.append(le.transform(vals))
        label_encoders[col] = le

    cat_tensor = torch.LongTensor(np.column_stack(cat_encoded))
    cat_dims = [len(le.classes_) for le in label_encoders.values()]

    # ── Numerical: scale ──────────────────────────────────────────────
    num_raw = np.array([
        [loan[col] for col in NUMERICAL_COLS]
        for loan in loans
    ], dtype=np.float32)

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_raw)
    num_tensor = torch.FloatTensor(num_scaled)

    return num_tensor, cat_tensor, cat_dims, label_encoders


# =====================================================================
# Demo runner
# =====================================================================

def demo_binary(num_t, cat_t, cat_dims):
    """Mode 1: Binary classification — P(default)."""
    print("\n" + "=" * 65)
    print("MODE 1: BINARY CLASSIFICATION")
    print("  Input  → loan features")
    print("  Output → P(default) ∈ [0, 1]")
    print("=" * 65)

    model = FlexTabTransformer(
        num_numerical_features=len(NUMERICAL_COLS),
        num_categorical_features=len(CATEGORICAL_COLS),
        categorical_dims=cat_dims,
        embedding_dim=32, depth=4, heads=4, dim_head=32,
        mlp_dim=256, dropout=0.2,
        output_mode='binary',
    )
    model.eval()

    with torch.no_grad():
        probs = model(num_t, cat_t)

    print(f"\n  {'Borrower':<30s} {'P(default)':>12s}  {'Prediction':>12s}")
    print("  " + "-" * 58)
    for i, loan in enumerate(SAMPLE_LOANS):
        p = probs[i].item()
        label = "DEFAULT" if p >= 0.5 else "GOOD"
        print(f"  {loan['name']:<30s} {p:>12.4f}  {label:>12s}")

    print(f"\n  Output shape: {probs.shape}  (batch_size,)")
    print("  Loss function: BCELoss (model applies sigmoid internally)")


def demo_multiclass(num_t, cat_t, cat_dims):
    """Mode 2: Multi-class — predict risk grade."""
    print("\n" + "=" * 65)
    print("MODE 2: MULTI-CLASS CLASSIFICATION")
    print("  Input  → loan features")
    print("  Output → risk grade probabilities (A/B/C/D/E)")
    print("=" * 65)

    model = FlexTabTransformer(
        num_numerical_features=len(NUMERICAL_COLS),
        num_categorical_features=len(CATEGORICAL_COLS),
        categorical_dims=cat_dims,
        embedding_dim=32, depth=4, heads=4, dim_head=32,
        mlp_dim=256, dropout=0.2,
        output_mode='multiclass',
        num_classes=5,
    )
    model.eval()

    with torch.no_grad():
        logits = model(num_t, cat_t)
        probs = torch.softmax(logits, dim=-1)

    print(f"\n  {'Borrower':<25s}", end="")
    for g in RISK_GRADES:
        print(f" {'P(' + g + ')':>8s}", end="")
    print(f"  {'Predicted':>10s}")
    print("  " + "-" * 75)

    for i, loan in enumerate(SAMPLE_LOANS):
        print(f"  {loan['name']:<25s}", end="")
        for j in range(5):
            print(f" {probs[i][j].item():>8.3f}", end="")
        pred_idx = probs[i].argmax().item()
        print(f"  {RISK_GRADES[pred_idx]:>10s}")

    print(f"\n  Output shape: {logits.shape}  (batch_size, num_classes)")
    print("  Loss function: CrossEntropyLoss (model outputs raw logits)")


def demo_regression(num_t, cat_t, cat_dims):
    """Mode 3: Regression — predict loss amount."""
    print("\n" + "=" * 65)
    print("MODE 3: REGRESSION")
    print("  Input  → loan features")
    print("  Output → predicted loss-given-default ($)")
    print("=" * 65)

    model = FlexTabTransformer(
        num_numerical_features=len(NUMERICAL_COLS),
        num_categorical_features=len(CATEGORICAL_COLS),
        categorical_dims=cat_dims,
        embedding_dim=32, depth=4, heads=4, dim_head=32,
        mlp_dim=256, dropout=0.2,
        output_mode='regression',
    )
    model.eval()

    with torch.no_grad():
        predictions = model(num_t, cat_t)

    print(f"\n  {'Borrower':<30s} {'Predicted Loss':>15s}")
    print("  " + "-" * 48)
    for i, loan in enumerate(SAMPLE_LOANS):
        val = predictions[i].item()
        print(f"  {loan['name']:<30s} ${abs(val)*1000:>13,.2f}")

    print(f"\n  Output shape: {predictions.shape}  (batch_size, 1)")
    print("  Loss function: MSELoss or HuberLoss (raw output, no activation)")
    print("  Note: values are from an untrained model — shown for shape demo")


def demo_multi_output(num_t, cat_t, cat_dims):
    """Mode 4: Multi-output — predict everything at once."""
    print("\n" + "=" * 65)
    print("MODE 4: MULTI-OUTPUT (sequence-to-output)")
    print("  Input  → loan features")
    print("  Output → {default_prob, loss_amount, risk_grade} simultaneously")
    print("=" * 65)

    model = FlexTabTransformer(
        num_numerical_features=len(NUMERICAL_COLS),
        num_categorical_features=len(CATEGORICAL_COLS),
        categorical_dims=cat_dims,
        embedding_dim=32, depth=4, heads=4, dim_head=32,
        mlp_dim=256, dropout=0.2,
        output_mode='multi_output',
        output_dims=[
            ('default_prob', 1),    # binary → apply sigmoid + BCELoss
            ('loss_amount', 1),     # regression → MSELoss
            ('risk_grade', 5),      # 5-class → CrossEntropyLoss
        ],
    )
    model.eval()

    with torch.no_grad():
        outputs = model(num_t, cat_t)

    # Process each head's output
    default_probs = torch.sigmoid(outputs['default_prob']).squeeze(-1)
    loss_amounts = outputs['loss_amount'].squeeze(-1)
    grade_probs = torch.softmax(outputs['risk_grade'], dim=-1)

    print(f"\n  Output keys: {list(outputs.keys())}")
    print(f"  Output shapes:")
    for k, v in outputs.items():
        print(f"    {k:20s} → {v.shape}")

    print(f"\n  {'Borrower':<25s} {'P(default)':>11s} {'Loss($)':>10s} {'Grade':>7s}")
    print("  " + "-" * 58)
    for i, loan in enumerate(SAMPLE_LOANS):
        p_def = default_probs[i].item()
        loss = abs(loss_amounts[i].item()) * 1000
        grade_idx = grade_probs[i].argmax().item()
        print(f"  {loan['name']:<25s} {p_def:>11.4f} ${loss:>8,.0f} {RISK_GRADES[grade_idx]:>7s}")

    print("\n  Loss functions (one per head, summed during training):")
    print("    default_prob → BCEWithLogitsLoss (apply sigmoid to raw output)")
    print("    loss_amount  → MSELoss or HuberLoss")
    print("    risk_grade   → CrossEntropyLoss")
    print("    Total loss   = w1*L_default + w2*L_loss + w3*L_grade")


def demo_ft_transformer(num_t, cat_t, cat_dims):
    """Bonus: FT-Transformer variant (numerical features as tokens)."""
    print("\n" + "=" * 65)
    print("BONUS: FT-TRANSFORMER VARIANT")
    print("  project_numerical=True → numericals become transformer tokens")
    print(f"  Sequence length: {len(CATEGORICAL_COLS)} cat + {len(NUMERICAL_COLS)} num"
          f" = {len(CATEGORICAL_COLS) + len(NUMERICAL_COLS)} tokens")
    print("=" * 65)

    model = FlexTabTransformer(
        num_numerical_features=len(NUMERICAL_COLS),
        num_categorical_features=len(CATEGORICAL_COLS),
        categorical_dims=cat_dims,
        embedding_dim=32, depth=4, heads=4, dim_head=32,
        mlp_dim=256, dropout=0.2,
        output_mode='binary',
        project_numerical=True,   # <-- the key difference
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        probs = model(num_t, cat_t)

    print(f"\n  Parameters: {n_params:,}")
    print(f"  {'Borrower':<30s} {'P(default)':>12s}")
    print("  " + "-" * 44)
    for i, loan in enumerate(SAMPLE_LOANS):
        print(f"  {loan['name']:<30s} {probs[i].item():>12.4f}")

    print("\n  With project_numerical=True, ALL features attend to each other")
    print("  through the transformer — no bypass path. This can capture")
    print("  interactions like (FICO_score <-> int_rate <-> home_ownership).")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 65)
    print("FlexTabTransformer Demo")
    print("3 sample borrowers × 5 output modes")
    print("=" * 65)

    # ── Prepare inputs ────────────────────────────────────────────────
    print("\nSample loan applications:")
    for loan in SAMPLE_LOANS:
        print(f"  • {loan['name']}")
        print(f"    term={loan['term']}, rate={loan['int_rate']}%, "
              f"FICO={loan['last_fico_range_high']}, "
              f"home={loan['home_ownership']}, "
              f"purpose={loan['purpose']}")

    num_t, cat_t, cat_dims, _ = prepare_inputs(SAMPLE_LOANS)

    print(f"\nTensor shapes:")
    print(f"  Numerical  : {num_t.shape}  ({len(NUMERICAL_COLS)} features)")
    print(f"  Categorical: {cat_t.shape}  ({len(CATEGORICAL_COLS)} features)")
    print(f"  Cat dims   : {cat_dims}")

    # ── Run all demos ─────────────────────────────────────────────────
    demo_binary(num_t, cat_t, cat_dims)
    demo_multiclass(num_t, cat_t, cat_dims)
    demo_regression(num_t, cat_t, cat_dims)
    demo_multi_output(num_t, cat_t, cat_dims)
    demo_ft_transformer(num_t, cat_t, cat_dims)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY: Input → Output mapping")
    print("=" * 65)
    print("""
  All modes share the SAME input format:
    numerical_features  : FloatTensor (batch, 33)
    categorical_features: LongTensor  (batch, 7)

  Output varies by mode:
    binary       → (batch,)     P(default) after sigmoid
    multiclass   → (batch, K)   raw logits → softmax for probabilities
    regression   → (batch, 1)   continuous value (no activation)
    multi_output → dict of tensors, one per head

  Architecture: encoder-only (no decoder needed for tabular data)
    Categorical tokens → Transformer self-attention → Pool
    Numerical features → bypass (or projected as tokens if FT-Transformer)
    Fused → task-specific MLP head(s)

  Note: outputs above are from UNTRAINED models (random weights).
  To get meaningful predictions, train with your labelled data using
  the appropriate loss function for each output mode.
""")


if __name__ == "__main__":
    main()
