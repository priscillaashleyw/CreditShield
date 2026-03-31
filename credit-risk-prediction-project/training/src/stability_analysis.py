import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, auc


def calculate_pr_auc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def safe_precision(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return tp / max(tp + fp, 1)


def safe_recall(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return tp / max(tp + fn, 1)


def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {
        "n_samples": len(y_true),
        "default_rate": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": safe_precision(y_true, y_pred),
        "recall": safe_recall(y_true, y_pred),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["auc_pr"] = calculate_pr_auc(y_true, y_prob)
    else:
        metrics["auc_roc"] = np.nan
        metrics["auc_pr"] = np.nan

    return metrics

def add_issue_year(df, issue_date_col="issue_d"):
    df = df.copy()
    df["issue_year"] = pd.to_datetime(df[issue_date_col], errors="coerce").dt.year
    return df


def make_income_buckets(df, income_col="annual_inc", n_buckets=4):
    df = df.copy()
    df["income_bucket"] = pd.qcut(
        df[income_col],
        q=n_buckets,
        duplicates="drop",
        labels=[f"Q{i+1}" for i in range(n_buckets)]
    )
    return df


def make_loan_amount_buckets(df, loan_col="loan_amnt", n_buckets=4):
    df = df.copy()
    df["loan_amount_bucket"] = pd.qcut(
        df[loan_col],
        q=n_buckets,
        duplicates="drop",
        labels=[f"Q{i+1}" for i in range(n_buckets)]
    )
    return df


def evaluate_by_segment(df, segment_col):
    rows = []

    for segment_value, group in df.groupby(segment_col, dropna=False):
        if len(group) == 0:
            continue

        metrics = calculate_metrics(
            y_true=group["y_true"].values,
            y_pred=group["y_pred"].values,
            y_prob=group["y_pred_proba"].values,
        )

        metrics[segment_col] = segment_value
        rows.append(metrics)

    result = pd.DataFrame(rows)
    cols = [segment_col] + [c for c in result.columns if c != segment_col]
    return result[cols].sort_values(by=segment_col).reset_index(drop=True)


def run_stability_analysis(
    eval_df,
    issue_date_col="issue_date",
    income_col="annual_inc",
    loan_col="loan_amnt",
):
    df = eval_df.copy()

    df = add_issue_year(df, issue_date_col=issue_date_col)
    df = make_income_buckets(df, income_col=income_col)
    df = make_loan_amount_buckets(df, loan_col=loan_col)

    outputs = {}

    if "issue_year" in df.columns:
        outputs["by_year"] = evaluate_by_segment(df, "issue_year")

    if "income_bucket" in df.columns:
        outputs["by_income_bucket"] = evaluate_by_segment(df, "income_bucket")

    if "loan_amount_bucket" in df.columns:
        outputs["by_loan_amount_bucket"] = evaluate_by_segment(df, "loan_amount_bucket")

    return outputs