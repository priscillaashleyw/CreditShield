import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RiskSegmentationAnalyzer:
    def __init__(
        self,
        low_cutoff=0.15,
        medium_cutoff=0.35,
        high_cutoff=0.55,
        review_conversion_rate=0.5,
        conditional_conversion_rate=0.7,
        opportunity_cost_factor=0.0
    ):
        """
        Parameters
        ----------
        low_cutoff : float
            Upper threshold for Low risk band
        medium_cutoff : float
            Upper threshold for Medium risk band
        high_cutoff : float
            Upper threshold for High risk band
        review_conversion_rate : float
            Fraction of manual review loans assumed to eventually be approved
        conditional_conversion_rate : float
            Fraction of conditional approve loans assumed to eventually be approved
        opportunity_cost_factor : float
            If rejecting a good borrower, add missed interest * factor as opportunity cost.
            Use 0.0 if you do not want to include opportunity cost.
        """
        self.low_cutoff = low_cutoff
        self.medium_cutoff = medium_cutoff
        self.high_cutoff = high_cutoff
        self.review_conversion_rate = review_conversion_rate
        self.conditional_conversion_rate = conditional_conversion_rate
        self.opportunity_cost_factor = opportunity_cost_factor

    def assign_risk_band(self, prob):
        if prob < self.low_cutoff:
            return "Low risk"
        elif prob < self.medium_cutoff:
            return "Medium risk"
        elif prob < self.high_cutoff:
            return "High risk"
        else:
            return "Very high risk"

    def assign_decision(self, prob):
        if prob < self.low_cutoff:
            return "Auto approve"
        elif prob < self.medium_cutoff:
            return "Manual review"
        elif prob < self.high_cutoff:
            return "Conditional approve"
        else:
            return "Reject"

    def build_segmentation_table(self, y_true, y_prob, X_test=None):
        """
        Build borrower-level segmentation output.
        y_true: actual labels (0=non-default, 1=default)
        y_prob: predicted default probabilities
        X_test: optional DataFrame of test features
        """
        if X_test is not None:
            df = X_test.copy().reset_index(drop=True)
        else:
            df = pd.DataFrame()

        df["actual_default"] = np.asarray(y_true).reshape(-1)
        df["pred_prob_default"] = np.asarray(y_prob).reshape(-1)
        df["risk_band"] = df["pred_prob_default"].apply(self.assign_risk_band)
        df["decision"] = df["pred_prob_default"].apply(self.assign_decision)

        band_order = ["Low risk", "Medium risk", "High risk", "Very high risk"]
        df["risk_band"] = pd.Categorical(df["risk_band"], categories=band_order, ordered=True)
        return df.sort_values("risk_band").reset_index(drop=True)

    def summarize_by_band(self, segmented_df):
        summary = (
            segmented_df.groupby(["risk_band", "decision"], observed=False)
            .agg(
                n_loans=("actual_default", "count"),
                actual_defaults=("actual_default", "sum"),
                avg_pred_prob=("pred_prob_default", "mean")
            )
            .reset_index()
        )
        summary["observed_default_rate"] = summary["actual_defaults"] / summary["n_loans"]
        return summary

    def summarize_band_only(self, segmented_df):
        summary = (
            segmented_df.groupby("risk_band", observed=False)
            .agg(
                n_loans=("actual_default", "count"),
                actual_defaults=("actual_default", "sum"),
                avg_pred_prob=("pred_prob_default", "mean")
            )
            .reset_index()
        )
        summary["observed_default_rate"] = summary["actual_defaults"] / summary["n_loans"]
        return summary

    def _approval_probability_from_decision(self, decision):
        if decision == "Auto approve":
            return 1.0
        elif decision == "Manual review":
            return self.review_conversion_rate
        elif decision == "Conditional approve":
            return self.conditional_conversion_rate
        else:
            return 0.0

    def add_profit_analysis(self, segmented_df, loan_amount_col="loan_amnt", interest_rate_col="int_rate"):
        """
        Adds expected profit based on business logic:
        - If approved and non-default -> gain interest
        - If approved and default -> lose full principal
        - If rejected and non-default -> optional opportunity cost
        Assumes:
        - y=0 means good loan
        - y=1 means default
        """
        df = segmented_df.copy()

        if loan_amount_col not in df.columns:
            raise ValueError(f"Column '{loan_amount_col}' not found in segmented_df")
        if interest_rate_col not in df.columns:
            raise ValueError(f"Column '{interest_rate_col}' not found in segmented_df")

        # Convert interest to decimal if likely stored as percentage
        rate = df[interest_rate_col].astype(float).copy()
        if rate.max() > 1:
            rate = rate / 100.0

        df["approval_probability"] = df["decision"].apply(self._approval_probability_from_decision)

        # gain if approved and borrower does not default
        df["interest_gain_if_good"] = df[loan_amount_col].astype(float) * rate

        # expected profit from actual realized outcomes
        df["expected_profit"] = np.where(
            df["actual_default"] == 0,
            df["approval_probability"] * df["interest_gain_if_good"],
            -df["approval_probability"] * df[loan_amount_col].astype(float)
        )

        # optional opportunity cost for rejected / not-approved good borrowers
        df["opportunity_cost"] = np.where(
            df["actual_default"] == 0,
            (1 - df["approval_probability"]) * df["interest_gain_if_good"] * self.opportunity_cost_factor,
            0.0
        )

        df["net_expected_profit"] = df["expected_profit"] - df["opportunity_cost"]

        return df

    def profit_summary(self, profit_df):
        summary = (
            profit_df.groupby(["risk_band", "decision"], observed=False)
            .agg(
                n_loans=("actual_default", "count"),
                total_expected_profit=("net_expected_profit", "sum"),
                avg_expected_profit=("net_expected_profit", "mean")
            )
            .reset_index()
        )
        return summary

    def plot_risk_band_distribution(self, segmented_df):
        band_counts = segmented_df["risk_band"].value_counts().sort_index()
        plt.figure(figsize=(8, 5))
        plt.bar(band_counts.index.astype(str), band_counts.values)
        plt.title("Loan Count by Risk Band")
        plt.xlabel("Risk Band")
        plt.ylabel("Number of Loans")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()

    def plot_default_rate_by_band(self, segmented_df):
        band_summary = self.summarize_band_only(segmented_df)
        plt.figure(figsize=(8, 5))
        plt.bar(band_summary["risk_band"].astype(str), band_summary["observed_default_rate"])
        plt.title("Observed Default Rate by Risk Band")
        plt.xlabel("Risk Band")
        plt.ylabel("Default Rate")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.show()


def run_risk_segmentation_analysis(
    y_true,
    y_prob,
    X_test=None,
    loan_amount_col="loan_amnt",
    interest_rate_col="int_rate",
    review_conversion_rate=0.5,
    conditional_conversion_rate=0.7,
    opportunity_cost_factor=0.0,
    make_plots=True
):
    analyzer = RiskSegmentationAnalyzer(
        review_conversion_rate=review_conversion_rate,
        conditional_conversion_rate=conditional_conversion_rate,
        opportunity_cost_factor=opportunity_cost_factor
    )

    segmented_df = analyzer.build_segmentation_table(y_true, y_prob, X_test)
    summary_by_band = analyzer.summarize_by_band(segmented_df)
    band_only_summary = analyzer.summarize_band_only(segmented_df)

    print("\n" + "=" * 80)
    print("RISK SEGMENTATION SUMMARY")
    print("=" * 80)
    print(summary_by_band.to_string(index=False))

    print("\n" + "=" * 80)
    print("BAND-ONLY PERFORMANCE SUMMARY")
    print("=" * 80)
    print(band_only_summary.to_string(index=False))

    profit_df = None
    profit_summary = None

    if X_test is not None and loan_amount_col in segmented_df.columns and interest_rate_col in segmented_df.columns:
        profit_df = analyzer.add_profit_analysis(
            segmented_df,
            loan_amount_col=loan_amount_col,
            interest_rate_col=interest_rate_col
        )
        profit_summary = analyzer.profit_summary(profit_df)

        print("\n" + "=" * 80)
        print("PROFIT SUMMARY")
        print("=" * 80)
        print(profit_summary.to_string(index=False))

    if make_plots:
        analyzer.plot_risk_band_distribution(segmented_df)
        analyzer.plot_default_rate_by_band(segmented_df)

    return segmented_df, summary_by_band, band_only_summary, profit_df, profit_summary