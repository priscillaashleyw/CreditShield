import os
import json
from typing import Dict, Any, List, TypedDict, Annotated

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from predictor import CreditRiskPredictor

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CreditShield — Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
)


# ── Predictor ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_predictor() -> CreditRiskPredictor:
    return CreditRiskPredictor("model_artifacts")


predictor = get_predictor()


# ── Constants ─────────────────────────────────────────────────────────────────

US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
]

EMP_LENGTH_OPTIONS = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years",
]

PURPOSE_OPTIONS = {
    "Debt Consolidation":      "debt_consolidation",
    "Credit Card Refinancing": "credit_card",
    "Home Improvement":        "home_improvement",
    "Major Purchase":          "major_purchase",
    "Medical":                 "medical",
    "Car":                     "car",
    "Small Business":          "small_business",
    "Wedding":                 "wedding",
    "Vacation":                "vacation",
    "Moving & Relocation":     "moving",
    "Other":                   "other",
}
PURPOSE_VALUES_TO_LABELS = {v: k for k, v in PURPOSE_OPTIONS.items()}

HOME_OWNERSHIP_OPTIONS  = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION_OPTIONS    = ["Not Verified", "Source Verified", "Verified"]
GRADE_OPTIONS           = ["A", "B", "C", "D", "E", "F", "G"]

# Default profile — None for FICO means "let the model estimate it"
FEATURE_DEFAULTS: Dict[str, Any] = {
    # ── Tier 1 — Basic Info ──────────────────────────────────────────────────
    "loan_amnt":              15000,
    "purpose":                "debt_consolidation",
    "title":                  "",
    "int_rate":               12.5,
    "annual_inc":             75000,
    "emp_length":             "5 years",
    "monthly_debt_payments":  1156,   # ≈ 18.5% of 75000/12
    "home_ownership":         "RENT",
    "addr_state":          "CA",
    "verification_status": "Verified",
    # ── Tier 2 — Credit Profile ──────────────────────────────────────────────
    "revol_util":              45.0,
    "revol_bal":               5000,
    "total_bc_limit":          20000,
    "total_acc":               25,
    "total_bal_ex_mort":       30000,
    "avg_cur_bal":             2500,
    "last_fico_range_low":     None,   # None → estimated by model
    "last_fico_range_high":    None,
    "pct_tl_nvr_dlq":          95.0,
    "years_since_earliest_cr": 10.0,
    "mo_sin_old_il_acct":      60,
    "mo_sin_old_rev_tl_op":    48,
    "mo_sin_rcnt_rev_tl_op":   12,
    "mths_since_recent_bc":    6,
    "mths_since_recent_inq":   3,
    # ── Hidden (never shown) ─────────────────────────────────────────────────
    "delinq_2yrs": 0,
}

EXAMPLES: Dict[str, Dict[str, Any]] = {
    "Low Risk": {
        "loan_amnt": 10000, "int_rate": 8.5,
        "emp_length": "10+ years", "annual_inc": 120000, "monthly_debt_payments": 1200,
        "home_ownership": "OWN", "addr_state": "CA",
        "verification_status": "Verified",
        "purpose": "debt_consolidation", "title": "Debt consolidation",
        "revol_util": 30.0, "revol_bal": 3000, "total_bc_limit": 15000,
        "total_acc": 20, "total_bal_ex_mort": 25000, "avg_cur_bal": 3000,
        "last_fico_range_low": 720, "last_fico_range_high": 724,
        "pct_tl_nvr_dlq": 98.0, "years_since_earliest_cr": 15.0,
        "mo_sin_old_il_acct": 120, "mo_sin_old_rev_tl_op": 96,
        "mo_sin_rcnt_rev_tl_op": 24, "mths_since_recent_bc": 12,
        "mths_since_recent_inq": 6, "delinq_2yrs": 0,
    },
    "High Risk": {
        "loan_amnt": 35000, "int_rate": 25.0,
        "emp_length": "< 1 year", "annual_inc": 30000, "monthly_debt_payments": 875,
        "home_ownership": "RENT", "addr_state": "NV",
        "verification_status": "Not Verified",
        "purpose": "credit_card", "title": "Credit card payoff",
        "revol_util": 95.0, "revol_bal": 20000, "total_bc_limit": 5000,
        "total_acc": 40, "total_bal_ex_mort": 10000, "avg_cur_bal": 1000,
        "last_fico_range_low": 580, "last_fico_range_high": 584,
        "pct_tl_nvr_dlq": 60.0, "years_since_earliest_cr": 2.0,
        "mo_sin_old_il_acct": 6, "mo_sin_old_rev_tl_op": 12,
        "mo_sin_rcnt_rev_tl_op": 1, "mths_since_recent_bc": 1,
        "mths_since_recent_inq": 1, "delinq_2yrs": 0,
    },
    "Borderline": {
        "loan_amnt": 20000, "int_rate": 15.0,
        "emp_length": "3 years", "annual_inc": 55000, "monthly_debt_payments": 1008,
        "home_ownership": "MORTGAGE", "addr_state": "TX",
        "verification_status": "Source Verified",
        "purpose": "home_improvement", "title": "Home renovation loan",
        "revol_util": 75.0, "revol_bal": 10000, "total_bc_limit": 10000,
        "total_acc": 30, "total_bal_ex_mort": 20000, "avg_cur_bal": 2000,
        "last_fico_range_low": 650, "last_fico_range_high": 654,
        "pct_tl_nvr_dlq": 85.0, "years_since_earliest_cr": 5.0,
        "mo_sin_old_il_acct": 36, "mo_sin_old_rev_tl_op": 48,
        "mo_sin_rcnt_rev_tl_op": 6, "mths_since_recent_bc": 6,
        "mths_since_recent_inq": 3, "delinq_2yrs": 0,
    },
}


# ── Session state ─────────────────────────────────────────────────────────────

def ensure_state() -> None:
    if "profile" not in st.session_state:
        st.session_state.profile = dict(FEATURE_DEFAULTS)
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_profile" not in st.session_state:
        st.session_state.agent_profile = dict(FEATURE_DEFAULTS)


ensure_state()


# ── Widget sync ───────────────────────────────────────────────────────────────

# Maps _num() widget key → (profile_key, slider_min, slider_max)
# widget key MUST match the key= argument passed to _num() in the form below
_WIDGET_MAP = {
    # Tier-1
    "loan_amnt":             ("loan_amnt",              1_000,   40_000),   # LC max = $40K
    "int_rate":              ("int_rate",                5.0,     31.0),    # LC max = 31%
    "annual_inc":            ("annual_inc",              10_000,  500_000), # p99 ≈ $260K; extend for comfort
    "monthly_debt_payments": ("monthly_debt_payments",   0,       15_000),  # 40% DTI on $450K income = $15K
    # Tier-2 credit usage
    "revol_util":            ("revol_util",              0.0,     100.0),
    "revol_bal":             ("revol_bal",               0,       150_000), # p99 ≈ $103K
    "total_bc_limit":        ("total_bc_limit",          0,       200_000), # p99 ≈ $109K
    "total_acc":             ("total_acc",               0,       100),     # p99 ≈ 61; some go to 169
    "total_bal_ex_mort":     ("total_bal_ex_mort",       0,       300_000), # p95 = $139K — old max was below p95!
    "avg_cur_bal":           ("avg_cur_bal",             0,       100_000), # p95 = $42K, p99 = $72K
    "pct_tl_nvr_dlq":        ("pct_tl_nvr_dlq",         0.0,     100.0),
    # Tier-2 credit history
    "years_cr":              ("years_since_earliest_cr", 0.0,     50.0),    # p99 ≈ 40 yr
    "mo_old_il":             ("mo_sin_old_il_acct",      0,       480),     # p99 ≈ 276; extend to 480 (40 yr)
    "mo_old_rev":            ("mo_sin_old_rev_tl_op",    0,       600),     # p99 ≈ 480 — old max was below p95!
    "mo_rcnt_rev":           ("mo_sin_rcnt_rev_tl_op",   0,       120),     # p99 ≈ 91
    "mths_bc":               ("mths_since_recent_bc",    0,       200),     # p99 ≈ 153
    "mths_inq":              ("mths_since_recent_inq",   0,       25),      # max = 25
    "fico_low":              ("last_fico_range_low",     300,     850),
}


def _apply_profile(profile: Dict[str, Any]) -> None:
    """
    Load a profile into session state AND all widget keys so sliders
    immediately reflect the new values on next render.
    """
    st.session_state.profile = dict(profile)
    st.session_state.prediction_result = None

    for widget_key, (profile_key, lo, hi) in _WIDGET_MAP.items():
        val = profile.get(profile_key)
        if val is None:
            st.session_state[f"_ti_{widget_key}"] = ""
            continue
        fval = float(val)
        # Clamp slider to its range; put out-of-range values in the text box
        clamped = float(min(max(fval, lo), hi))
        st.session_state[f"_sl_{widget_key}"] = clamped
        st.session_state[f"_ti_{widget_key}"] = "" if lo <= fval <= hi else str(fval)

    # FICO checkbox
    st.session_state["know_fico_cb"] = profile.get("last_fico_range_low") is not None


# ── UI helpers ────────────────────────────────────────────────────────────────

def _num(
    label: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float,
    help_text: str,
    key: str,
) -> float:
    """
    Renders a labelled slider (wide) with a compact text input (narrow) to
    its right.  The text input shows placeholder ghost text and overrides
    the slider when filled.  Returns the slider value when the box is empty.
    """
    safe_default = float(min(max(float(default), float(min_val)), float(max_val)))

    sc, tc = st.columns([5, 1])
    with sc:
        slider_val = st.slider(
            label,
            float(min_val), float(max_val), safe_default, float(step),
            help=help_text,
            key=f"_sl_{key}",
        )
    with tc:
        # collapsed removes label space; margin nudges the box down so it
        # sits level with the slider track, not the label row above it.
        st.markdown('<div style="margin-top:1.65rem"></div>', unsafe_allow_html=True)
        custom_text: str = st.text_input(
            label,
            placeholder="Custom",
            label_visibility="collapsed",
            key=f"_ti_{key}",
        )

    if custom_text.strip():
        try:
            return float(custom_text.strip())
        except ValueError:
            pass   # invalid input — fall through to slider value

    return float(slider_val)


def _render_credit_fields(p: Dict[str, Any]) -> tuple:
    """
    Renders all credit-detail sliders (credit card balances, debts, history).
    Works whether called directly or inside an st.expander context.
    Returns a tuple of all captured values for use in the profile dict.
    """
    # ── Credit cards ──────────────────────────────────────────────────────────
    st.markdown("##### 💳 Credit Cards")
    cc1, cc2 = st.columns(2)
    with cc1:
        revol_util = _num(
            "Credit Utilization (%)", 0.0, 100.0, p.get("revol_util", 45.0), 1.0,
            "Outstanding balance ÷ total credit limit × 100  —  NOT your monthly spending. "
            "Example: $3,000 owed on a $10,000 total limit = 30%. "
            "Find it on Credit Karma or your credit card statements.",
            "revol_util",
        )
        revol_bal = _num(
            "Total Outstanding Balance ($)", 0, 150_000, p.get("revol_bal", 5000), 500,
            "Total amount you currently owe across all credit cards and revolving accounts",
            "revol_bal",
        )
    with cc2:
        total_bc_limit = _num(
            "Total Credit Card Limit ($)", 0, 200_000, p.get("total_bc_limit", 20000), 1_000,
            "Combined credit limit across all your credit cards",
            "total_bc_limit",
        )

    # ── Overall debt ──────────────────────────────────────────────────────────
    st.markdown("##### 🏦 Debts & Balances")
    od1, od2 = st.columns(2)
    with od1:
        total_acc = int(_num(
            "Total Credit Accounts Ever", 0, 100, p.get("total_acc", 25), 1,
            "Total number of credit accounts you've ever had (open or closed)",
            "total_acc",
        ))
        total_bal_ex_mort = _num(
            "Total Debt excl. Mortgage ($)", 0, 300_000, p.get("total_bal_ex_mort", 30000), 1_000,
            "Total amount owed on all loans and credit, not counting your mortgage",
            "total_bal_ex_mort",
        )
    with od2:
        avg_cur_bal = _num(
            "Average Account Balance ($)", 0, 100_000, p.get("avg_cur_bal", 2500), 500,
            "Average current balance across all your open accounts",
            "avg_cur_bal",
        )

    # ── Credit history ────────────────────────────────────────────────────────
    st.markdown("##### 📅 Credit History")
    ch1, ch2 = st.columns(2)
    with ch1:
        pct_tl_nvr_dlq = _num(
            "% of Accounts Never Late", 0.0, 100.0, p.get("pct_tl_nvr_dlq", 95.0), 1.0,
            "Percentage of your accounts that have never had a late payment",
            "pct_tl_nvr_dlq",
        )
        years_since_earliest_cr = _num(
            "Years Since First Credit Account", 0.0, 50.0,
            p.get("years_since_earliest_cr", 10.0), 0.5,
            "How many years since you first opened any credit account",
            "years_cr",
        )
        mo_sin_old_il_acct = int(_num(
            "Months Since Oldest Loan", 0, 480, p.get("mo_sin_old_il_acct", 60), 1,
            "How many months since your oldest loan was opened (car, student, personal, etc.)",
            "mo_old_il",
        ))
        mo_sin_old_rev_tl_op = int(_num(
            "Months Since Oldest Credit Card", 0, 600, p.get("mo_sin_old_rev_tl_op", 48), 1,
            "How many months since you first opened a credit card",
            "mo_old_rev",
        ))
    with ch2:
        mo_sin_rcnt_rev_tl_op = int(_num(
            "Months Since Newest Credit Card", 0, 120, p.get("mo_sin_rcnt_rev_tl_op", 12), 1,
            "How many months since you last opened a new credit card",
            "mo_rcnt_rev",
        ))
        mths_since_recent_bc = int(_num(
            "Months Since Newest Bank Card", 0, 200, p.get("mths_since_recent_bc", 6), 1,
            "How many months since you last opened or used a bank credit card",
            "mths_bc",
        ))
        mths_since_recent_inq = int(_num(
            "Months Since Last Credit Check", 0, 25, p.get("mths_since_recent_inq", 3), 1,
            "How many months since a lender last pulled your credit report",
            "mths_inq",
        ))

    return (
        revol_util, revol_bal, total_bc_limit, total_acc, total_bal_ex_mort,
        avg_cur_bal, pct_tl_nvr_dlq, years_since_earliest_cr,
        mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op,
        mths_since_recent_bc, mths_since_recent_inq,
    )


def _render_fico_fields(p: Dict[str, Any]) -> tuple:
    """
    Renders only the 7 fields that map directly to FICO's 5 factors.
    Used when the user doesn't know their credit score.
    Returns the same 13-value tuple as _render_credit_fields so the caller
    can unpack identically; the 6 hidden fields carry saved/default values.

    FICO factors covered:
      Payment History (35%): pct_tl_nvr_dlq
      Utilization      (30%): revol_util, revol_bal, total_bc_limit
      History Length   (15%): years_since_earliest_cr
      Credit Mix       (10%): total_acc
      New Credit       (10%): mths_since_recent_inq
    """
    ff1, ff2 = st.columns(2)

    with ff1:
        pct_tl_nvr_dlq = _num(
            "% of Accounts Never Late", 0.0, 100.0, p.get("pct_tl_nvr_dlq", 95.0), 1.0,
            "Percentage of your accounts that have never had a late payment  ·  Payment History (35% of FICO)",
            "pct_tl_nvr_dlq",
        )
        revol_util = _num(
            "Credit Utilization (%)", 0.0, 100.0, p.get("revol_util", 45.0), 1.0,
            "Outstanding balance ÷ total credit limit × 100. "
            "Example: $3,000 owed on a $10,000 limit = 30%  ·  Utilization (30% of FICO)",
            "revol_util",
        )
        revol_bal = _num(
            "Total Outstanding Balance ($)", 0, 150_000, p.get("revol_bal", 5000), 500,
            "Total owed across all credit cards and revolving accounts  ·  Utilization",
            "revol_bal",
        )
        total_bc_limit = _num(
            "Total Credit Card Limit ($)", 0, 200_000, p.get("total_bc_limit", 20000), 1_000,
            "Combined credit limit across all your credit cards  ·  Utilization",
            "total_bc_limit",
        )

    with ff2:
        years_since_earliest_cr = _num(
            "Years Since First Credit Account", 0.0, 50.0,
            p.get("years_since_earliest_cr", 10.0), 0.5,
            "How many years since you first opened any credit account  ·  History Length (15% of FICO)",
            "years_cr",
        )
        total_acc = int(_num(
            "Total Credit Accounts Ever", 0, 100, p.get("total_acc", 25), 1,
            "Total number of credit accounts you've ever had (open or closed)  ·  Credit Mix (10% of FICO)",
            "total_acc",
        ))
        mths_since_recent_inq = int(_num(
            "Months Since Last Credit Check", 0, 25, p.get("mths_since_recent_inq", 3), 1,
            "How many months since a lender last pulled your credit report  ·  New Credit (10% of FICO)",
            "mths_inq",
        ))

    # Remaining credit fields — adjustable but optional, collapsed by default
    with st.expander("📊 Additional credit details *(optional — improves accuracy)*", expanded=False):
        ax1, ax2 = st.columns(2)
        with ax1:
            total_bal_ex_mort = _num(
                "Total Debt excl. Mortgage ($)", 0, 300_000, p.get("total_bal_ex_mort", 30000), 1_000,
                "Total amount owed on all loans and credit, not counting your mortgage",
                "total_bal_ex_mort",
            )
            mo_sin_old_il_acct = int(_num(
                "Months Since Oldest Loan", 0, 480, p.get("mo_sin_old_il_acct", 60), 1,
                "How many months since your oldest loan was opened (car, student, personal, etc.)",
                "mo_old_il",
            ))
            mo_sin_old_rev_tl_op = int(_num(
                "Months Since Oldest Credit Card", 0, 600, p.get("mo_sin_old_rev_tl_op", 48), 1,
                "How many months since you first opened a credit card",
                "mo_old_rev",
            ))
        with ax2:
            avg_cur_bal = _num(
                "Average Account Balance ($)", 0, 100_000, p.get("avg_cur_bal", 2500), 500,
                "Average current balance across all your open accounts",
                "avg_cur_bal",
            )
            mo_sin_rcnt_rev_tl_op = int(_num(
                "Months Since Newest Credit Card", 0, 120, p.get("mo_sin_rcnt_rev_tl_op", 12), 1,
                "How many months since you last opened a new credit card",
                "mo_rcnt_rev",
            ))
            mths_since_recent_bc = int(_num(
                "Months Since Newest Bank Card", 0, 200, p.get("mths_since_recent_bc", 6), 1,
                "How many months since you last opened or used a bank credit card",
                "mths_bc",
            ))

    return (
        revol_util, revol_bal, total_bc_limit, total_acc, total_bal_ex_mort,
        avg_cur_bal, pct_tl_nvr_dlq, years_since_earliest_cr,
        mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op,
        mths_since_recent_bc, mths_since_recent_inq,
    )


# ── Visualisation ─────────────────────────────────────────────────────────────

def create_visualization(default_prob: float, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2))
    x = np.linspace(0, 1, 100)
    colors = plt.cm.RdYlGn_r(x)
    for i in range(len(x) - 1):
        ax.fill_between([x[i], x[i + 1]], 0, 1, color=colors[i], alpha=0.7)
    ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold ({threshold:.0%})")
    ax.plot(default_prob, 0.5, "ro", markersize=12,
            label=f"Prediction ({default_prob:.1%})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Default Probability")
    ax.set_title("Risk Assessment")
    ax.legend(loc="upper right")
    ax.set_yticks([])
    plt.tight_layout()
    return fig


# ── Profile → predictor input ─────────────────────────────────────────────────

def profile_to_predictor_input(profile: Dict[str, Any]) -> Dict[str, Any]:
    inp: Dict[str, Any] = {
        "loan_amnt":             float(profile["loan_amnt"]),
        "int_rate":              float(profile["int_rate"]),
        "emp_length":            str(profile["emp_length"]),
        "annual_inc":            float(profile["annual_inc"]),
        "dti":                   round(
            float(profile.get("monthly_debt_payments", 0))
            / max(float(profile["annual_inc"]) / 12.0, 1.0)
            * 100.0,
            2,
        ),
        "revol_util":            f"{float(profile['revol_util'])}%",
        "delinq_2yrs":           0,        # always default — not shown in UI
        "total_acc":             int(profile["total_acc"]),
        "revol_bal":             float(profile["revol_bal"]),
        "total_bc_limit":        float(profile["total_bc_limit"]),
        "total_bal_ex_mort":     float(profile["total_bal_ex_mort"]),
        "avg_cur_bal":           float(profile["avg_cur_bal"]),
        "mo_sin_old_il_acct":    float(profile["mo_sin_old_il_acct"]),
        "mo_sin_old_rev_tl_op":  float(profile["mo_sin_old_rev_tl_op"]),
        "mo_sin_rcnt_rev_tl_op": float(profile["mo_sin_rcnt_rev_tl_op"]),
        "mths_since_recent_bc":  float(profile["mths_since_recent_bc"]),
        "mths_since_recent_inq": float(profile["mths_since_recent_inq"]),
        "pct_tl_nvr_dlq":        float(profile["pct_tl_nvr_dlq"]) / 100.0,
        "years_since_earliest_cr": float(profile["years_since_earliest_cr"]),
        "addr_state":            str(profile["addr_state"]),
        "home_ownership":        str(profile["home_ownership"]),
        "purpose":               str(profile["purpose"]),
        "verification_status":   str(profile["verification_status"]),
        "title":                 str(profile["title"]),
    }

    # FICO — pass None to trigger the estimator in predictor.py
    fico_low = profile.get("last_fico_range_low")
    if fico_low is not None and float(fico_low) > 0:
        inp["last_fico_range_low"]  = float(fico_low)
        inp["last_fico_range_high"] = float(
            profile.get("last_fico_range_high") or float(fico_low) + 4
        )
    # else: omit both → predictor estimates from other Tier-2 fields

    return inp


def run_prediction(profile: Dict[str, Any]) -> Dict[str, Any]:
    return predictor.predict(profile_to_predictor_input(profile))


# ── LLM / Agent ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    current_profile: Dict[str, Any]


def get_llm():
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key     = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        return ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "CreditShield",
            },
        )
    if openai_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    return None


def _make_system_prompt(profile: Dict[str, Any]) -> str:
    key_fields = [
        "loan_amnt", "int_rate", "annual_inc",
        "monthly_debt_payments", "last_fico_range_low", "home_ownership", "purpose",
    ]
    summary = {k: profile.get(k) for k in key_fields}
    return f"""You are a helpful Financial Credit Risk Assistant with access to a what-if simulation tool.

Current profile summary:
{json.dumps(summary, indent=2)}

Available tools:
- predict_credit_risk: run a prediction, optionally changing features first
- get_current_profile: show all current profile values
- reset_profile: revert all values to the original defaults

Always summarise results clearly.
When comparing scenarios, highlight the change in probability.
Use exact feature names and valid categorical values as documented in predict_credit_risk.
Important: the model was trained on Lending Club data (2013-2014) with loan amounts up to $40,000.
If the user requests values far outside this range, warn that the prediction may be unreliable.
"""


@tool
def predict_credit_risk(
    updates: Dict[str, Any],
    profile: Annotated[Dict[str, Any], InjectedToolArg],
) -> str:
    """Run a credit risk prediction using the current profile, with optional field updates."""
    working = dict(profile)
    applied, unknown = [], []
    for k, v in updates.items():
        if k in working:
            working[k] = v
            applied.append(f"{k}={v}")
        else:
            unknown.append(k)
    try:
        result = run_prediction(working)
        if not result["success"]:
            return json.dumps({"error": result.get("error", "Prediction failed")})
        output = {
            "decision":            result["decision"],
            "default_probability": f"{result['default_probability']:.2%}",
            "risk_level":          result["risk_level"],
            "threshold":           f"{result['optimal_threshold']:.2%}",
            "fico_estimated":      result.get("fico_estimated", False),
            "updates_applied":     applied,
        }
        if unknown:
            output["unknown_features_ignored"] = unknown
        return json.dumps(output)
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {e}"})


@tool
def get_current_profile(profile: Annotated[Dict[str, Any], InjectedToolArg]) -> str:
    """Return the current borrower profile used by the chatbot."""
    return json.dumps(profile, indent=2)


@tool
def reset_profile(profile: Annotated[Dict[str, Any], InjectedToolArg]) -> str:
    """Reset the chatbot profile back to the default values."""
    return json.dumps({"status": "reset", "new_profile": FEATURE_DEFAULTS})


@st.cache_resource
def build_graph():
    llm = get_llm()
    if llm is None:
        return None
    tools = [predict_credit_risk, get_current_profile, reset_profile]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: AgentState) -> dict:
        profile     = state.get("current_profile", dict(FEATURE_DEFAULTS))
        non_system  = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        messages    = [SystemMessage(content=_make_system_prompt(profile))] + non_system
        response    = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        profile      = dict(state.get("current_profile", FEATURE_DEFAULTS))
        outputs      = []

        for tc in last_message.tool_calls:
            name = tc["name"]
            if name == "predict_credit_risk":
                raw     = tc["args"]
                updates = raw.get("updates") or {k: v for k, v in raw.items() if k in profile}
                for k, v in updates.items():
                    if k in profile:
                        profile[k] = v
                result = predict_credit_risk.invoke({"updates": updates, "profile": profile})
            elif name == "get_current_profile":
                result = get_current_profile.invoke({"profile": profile})
            elif name == "reset_profile":
                profile = dict(FEATURE_DEFAULTS)
                result  = reset_profile.invoke({"profile": profile})
            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})
            outputs.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=name)
            )

        return {"messages": outputs, "current_profile": profile}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", chatbot)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def ask_agent(user_input: str) -> str:
    graph = build_graph()
    if graph is None:
        return (
            "OpenRouter API key is not set. "
            "Please export OPENROUTER_API_KEY (or OPENAI_API_KEY) before launching Streamlit."
        )

    state: AgentState = {
        "messages":        st.session_state.chat_messages[:],
        "current_profile": dict(st.session_state.agent_profile),
    }
    state["messages"] = add_messages(state["messages"], [HumanMessage(content=user_input)])

    try:
        final = graph.invoke(state)
    except Exception as e:
        return f"LLM request failed: {e}"

    st.session_state.chat_messages = final["messages"]
    st.session_state.agent_profile = final["current_profile"]

    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "No response was returned by the model."


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("🏦 CreditShield — Credit Risk Predictor")
st.caption(
    "Fill in the loan application on the left, then click **Assess Credit Risk**. "
    "Use the AI assistant on the right to explore what-if scenarios."
)

left, right = st.columns([2, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — Application form
# ─────────────────────────────────────────────────────────────────────────────
with left:

    # Quick-load examples
    ex_c1, ex_c2, ex_c3, ex_c4, ex_c5 = st.columns(5)
    with ex_c1:
        st.markdown("**Quick load:**")
    for col, (name, _) in zip([ex_c2, ex_c3, ex_c4], EXAMPLES.items()):
        with col:
            if st.button(name, use_container_width=True, key=f"ex_{name}"):
                _apply_profile(EXAMPLES[name])
                st.rerun()
    with ex_c5:
        if st.button("🔄 Reset", use_container_width=True):
            _apply_profile(FEATURE_DEFAULTS)
            st.rerun()

    p = st.session_state.profile   # shorthand

    # ── TIER 1 ── Basic Info ─────────────────────────────────────────────────
    st.markdown("### 📋 Loan Details")
    ld1, ld2 = st.columns([3, 2])

    with ld1:
        loan_amnt = _num(
            "Loan Amount ($)", 1_000, 40_000, p["loan_amnt"], 500,
            "How much money you want to borrow", "loan_amnt",
        )
        int_rate = _num(
            "Interest Rate (%)", 5.0, 31.0, p["int_rate"], 0.1,
            "The yearly cost of borrowing, as a % of the loan", "int_rate",
        )

    with ld2:
        purpose_labels  = list(PURPOSE_OPTIONS.keys())
        purpose_values  = list(PURPOSE_OPTIONS.values())
        cur_purpose_val = p.get("purpose", "debt_consolidation")
        purpose_idx     = (
            purpose_values.index(cur_purpose_val)
            if cur_purpose_val in purpose_values else 0
        )
        purpose_label = st.selectbox(
            "Loan Purpose", purpose_labels, index=purpose_idx,
            help="What you plan to use the loan for",
        )
        purpose = PURPOSE_OPTIONS[purpose_label]

        title = st.text_input(
            "Describe your loan",
            value=p.get("title", ""),
            help="A short description of what the loan is for",
        )

    st.markdown("### 💼 Your Finances")
    yf1, yf2 = st.columns([3, 2])

    with yf1:
        annual_inc = _num(
            "Annual Income ($)", 10_000, 500_000, p["annual_inc"], 1_000,
            "Your total yearly income before taxes", "annual_inc",
        )
        emp_length = st.selectbox(
            "Employment duration", EMP_LENGTH_OPTIONS,
            index=EMP_LENGTH_OPTIONS.index(p.get("emp_length", "5 years")),
            help="How long you've been at your current job",
        )
        monthly_debt_payments = _num(
            "Monthly Debt Payments ($)", 0, 15_000, p.get("monthly_debt_payments", 0), 50,
            "Total monthly payments on all your current debts — car loan, student loan, credit cards, etc. "
            "(Your DTI ratio will be calculated automatically)",
            "monthly_debt_payments",
        )

    with yf2:
        home_ownership = st.selectbox(
            "Home Ownership", HOME_OWNERSHIP_OPTIONS,
            index=HOME_OWNERSHIP_OPTIONS.index(p.get("home_ownership", "RENT")),
            help="Whether you rent, have a mortgage, fully own your home, or other",
        )
        addr_state = st.selectbox(
            "State", US_STATES,
            index=US_STATES.index(p["addr_state"]) if p.get("addr_state") in US_STATES else 0,
            help="The US state you live in",
        )
        verification_status = st.selectbox(
            "Income Verification", VERIFICATION_OPTIONS,
            index=VERIFICATION_OPTIONS.index(p.get("verification_status", "Verified")),
            help="Whether your income has been verified by the lender",
        )
        st.caption(
            "ℹ️ Set by the lender after checking your documents — defaults to *Verified* "
            "since most approved Lending Club applications go through income verification."
        )

    # ── TIER 3 ── Credit Score ─────────────────────────────────────────────────
    st.markdown("### 🎯 Credit Score")
    st.caption(
        "Your FICO score alone accounts for over **56%** of the model's signal — it's the single "
        "most important input. If you don't know yours, fill in the credit details below and "
        "we'll estimate it for you."
    )

    know_fico = st.checkbox(
        "I know my credit score (FICO)",
        value=p.get("last_fico_range_low") is not None,
        key="know_fico_cb",
        help=(
            "FICO scores range 300–850: 580+ Fair · 670+ Good · 740+ Very Good · 800+ Exceptional. "
            "Check yours free on Credit Karma, your bank app, or credit card portal."
        ),
    )

    if know_fico:
        fico_default = p.get("last_fico_range_low") or 680
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            last_fico_range_low: Any = int(_num(
                "Your Credit Score (FICO)", 300, 850, fico_default, 5,
                "300–579 Poor · 580–669 Fair · 670–739 Good · 740–799 Very Good · 800–850 Exceptional",
                "fico_low",
            ))
            last_fico_range_high: Any = last_fico_range_low + 4
        with fc2:
            st.info(
                "**Credit score provided ✓**  \n"
                "The remaining ~44% of signal comes from credit details. "
                "Expand below to add them for a sharper result."
            )
        with st.expander("📊 Credit Details *(optional — improves accuracy)*", expanded=False):
            (revol_util, revol_bal, total_bc_limit, total_acc, total_bal_ex_mort,
             avg_cur_bal, pct_tl_nvr_dlq, years_since_earliest_cr,
             mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op,
             mths_since_recent_bc, mths_since_recent_inq) = _render_credit_fields(p)
    else:
        last_fico_range_low  = None
        last_fico_range_high = None
        st.info(
            "💡 **No credit score? No problem.**  \n"
            "Fill in the 7 fields below — they cover FICO's 5 scoring factors and we'll "
            "estimate your score automatically. An *Estimated* badge will appear on the result."
        )
        (revol_util, revol_bal, total_bc_limit, total_acc, total_bal_ex_mort,
         avg_cur_bal, pct_tl_nvr_dlq, years_since_earliest_cr,
         mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op,
         mths_since_recent_bc, mths_since_recent_inq) = _render_fico_fields(p)

    # Collect everything into the profile
    current_profile: Dict[str, Any] = {
        "loan_amnt":             loan_amnt,
        "int_rate":              int_rate,
        "emp_length":            emp_length,
        "annual_inc":            annual_inc,
        "monthly_debt_payments": monthly_debt_payments,
        "home_ownership":        home_ownership,
        "addr_state":            addr_state,
        "verification_status":   verification_status,
        "purpose":               purpose,
        "title":                 title,
        "revol_util":            revol_util,
        "revol_bal":             revol_bal,
        "total_bc_limit":        total_bc_limit,
        "total_acc":             total_acc,
        "total_bal_ex_mort":     total_bal_ex_mort,
        "avg_cur_bal":           avg_cur_bal,
        "last_fico_range_low":   last_fico_range_low,
        "last_fico_range_high":  last_fico_range_high,
        "pct_tl_nvr_dlq":        pct_tl_nvr_dlq,
        "years_since_earliest_cr": years_since_earliest_cr,
        "mo_sin_old_il_acct":    mo_sin_old_il_acct,
        "mo_sin_old_rev_tl_op":  mo_sin_old_rev_tl_op,
        "mo_sin_rcnt_rev_tl_op": mo_sin_rcnt_rev_tl_op,
        "mths_since_recent_bc":  mths_since_recent_bc,
        "mths_since_recent_inq": mths_since_recent_inq,
        "delinq_2yrs":           0,
    }
    st.session_state.profile = current_profile

    # Submit
    if st.button("🔍 Assess Credit Risk", use_container_width=True, type="primary"):
        with st.spinner("Running assessment..."):
            st.session_state.prediction_result = run_prediction(st.session_state.profile)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Assessment Results")
    result = st.session_state.prediction_result

    if result is None:
        st.info("Fill in the form above and click **Assess Credit Risk** to see the result.")

    elif not result.get("success"):
        st.error(f"⚠️ {result.get('error', 'Prediction failed')}")

    else:
        decision       = result["decision"]
        prob           = result["default_probability"]
        risk_level     = result["risk_level"]
        conf           = result["confidence"]
        threshold      = result["optimal_threshold"]
        fico_estimated = result.get("fico_estimated", False)
        fico_est_val   = result.get("fico_estimated_value")

        if decision == "APPROVE":
            st.success("✅  LOAN APPROVED")
        else:
            st.error("❌  LOAN REJECTED")

        if fico_estimated:
            st.warning(
                f"📊 **Credit score was estimated** (~{fico_est_val}) from the credit profile "
                f"fields you provided. For a more accurate result, tick *I know my credit score* "
                f"in the Credit Profile section and enter your actual score."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Default Probability", f"{prob:.2%}")
        m2.metric("Risk Level", risk_level)
        m3.metric("Model Confidence", f"{conf:.0%}")
        m4.metric("Decision Threshold", f"{threshold:.0%}")

        fig = create_visualization(prob, threshold)
        st.pyplot(fig, clear_figure=True)
        st.caption(
            f"Model uses {len(predictor.feature_list)} features  |  "
            f"Threshold optimised for profit  |  "
            f"{'⚠️ FICO estimated' if fico_estimated else '✓ FICO provided'}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — AI Assistant
# ─────────────────────────────────────────────────────────────────────────────
with right:
    st.subheader("🤖 AI Assistant")
    st.caption("Ask questions, explore what-if scenarios, or compare risk profiles.")

    key_present = bool(
        os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    if not key_present:
        st.warning(
            "No API key detected. Export your key before launching Streamlit:\n\n"
            "```\nexport OPENROUTER_API_KEY='your-key-here'\n```"
        )
    else:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.agent_profile = dict(FEATURE_DEFAULTS)
            st.rerun()

        for msg in st.session_state.chat_messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage) and msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        prompt = st.chat_input("Ask about risk, scenarios, or update features...")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = ask_agent(prompt)
                    st.markdown(reply)

    st.markdown("---")
    st.markdown("**Example prompts**")
    st.markdown(
        "- What is my current risk?\n"
        "- What if my income were $90,000 and DTI 10?\n"
        "- Reset my profile\n"
        "- Show my current profile\n"
        "- Compare grade C vs grade B\n"
        "- What factors matter most?"
    )
