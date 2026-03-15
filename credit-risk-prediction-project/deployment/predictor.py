# deployment/predictor.py
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _pick_latest(files):
    files = list(files)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


class CreditRiskPredictor:
    """Predictor using trained model artifacts and exported feature contract."""

    def __init__(self, model_dir="model_artifacts"):
        self.model_dir = Path(model_dir).resolve()
        self.deployment_dir = self.model_dir.parent
        self.project_root = self.deployment_dir.parent
        self.training_dir = self.project_root / "training"
        self.training_models_dir = self.training_dir / "models"

        self.model = None
        self.scaler = None
        self.imputer = None
        self.optimal_threshold = 0.28

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Ensure deployment artifacts are ready before loading
        self._sync_latest_artifacts()
        self._ensure_feature_json()

        self.feature_list = self._load_actual_features()
        print(f"📋 Using {len(self.feature_list)} ACTUAL features")

        self.base_features_needed = self._extract_base_features()
        print(f"📋 Expecting {len(self.base_features_needed)} base input features")

        self.load_artifacts()

    def _sync_latest_artifacts(self):
        """Copy latest training artifacts into deployment/model_artifacts with fixed names."""
        if not self.training_models_dir.exists():
            raise FileNotFoundError(f"Training models directory not found: {self.training_models_dir}")

        xgb_src = _pick_latest(self.training_models_dir.glob("xgb_optimized_*.pkl"))
        if xgb_src is None:
            xgb_src = _pick_latest(self.training_models_dir.glob("xgb_best_*.pkl"))

        scaler_src = _pick_latest(self.training_models_dir.glob("scaler_our_enhanced_*.pkl"))
        if scaler_src is None:
            scaler_src = _pick_latest(self.training_models_dir.glob("scaler_paper_16_*.pkl"))

        imputer_src = _pick_latest(self.training_models_dir.glob("imputer_our_enhanced_*.pkl"))
        if imputer_src is None:
            imputer_src = _pick_latest(self.training_models_dir.glob("imputer_paper_16_*.pkl"))

        required = {
            "xgb_best_model.pkl": xgb_src,
            "scaler.pkl": scaler_src,
            "imputer.pkl": imputer_src,
        }

        print("🔄 Syncing latest artifacts from training/models ...")
        for dst_name, src_path in required.items():
            if src_path is None:
                raise FileNotFoundError(
                    f"Could not find source artifact for {dst_name} in {self.training_models_dir}"
                )

            dst_path = self.model_dir / dst_name

            print(f"   {src_path.name}  ->  {dst_name}")
            print(f"      source size: {src_path.stat().st_size:,} bytes")

            shutil.copyfile(src_path, dst_path)

            print(f"      copied size: {dst_path.stat().st_size:,} bytes")

            if dst_path.stat().st_size < 1024:
                raise ValueError(
                    f"Copied artifact {dst_path} is suspiciously small "
                    f"({dst_path.stat().st_size} bytes)."
                )

        print("✅ Synced latest model artifacts to deployment/model_artifacts")

    def _ensure_feature_json(self):
        """Ensure training_features.json exists; generate it from training/find_features.py if missing."""
        json_path = self.model_dir / "training_features.json"
        if json_path.exists():
            return

        script_path = self.training_dir / "find_features.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"training_features.json missing and generator script not found: {script_path}"
            )

        print("⚠️ training_features.json missing. Generating from training/find_features.py ...")
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(self.training_dir),
            check=True,
        )

        if not json_path.exists():
            raise FileNotFoundError(
                f"Feature JSON was not created successfully at: {json_path}"
            )

        print("✅ Generated training_features.json")

    def _load_actual_features(self):
        """Load the actual features used in training."""
        feature_file = self.model_dir / "training_features.json"
        if not feature_file.exists():
            print(f"⚠️ {feature_file} not found")
            return []

        with open(feature_file, "r") as f:
            data = json.load(f)

        if "feature_names" in data and isinstance(data["feature_names"], list):
            return data["feature_names"]
        elif "enhanced_features" in data and isinstance(data["enhanced_features"], list):
            return data["enhanced_features"]

        print(f"❌ Could not find feature list in JSON. Keys: {list(data.keys())}")
        return []

    def _extract_base_features(self):
        """Extract base user-input features from engineered feature list."""
        if not self.feature_list:
            return []

        base_features = set()
        for feature in self.feature_list:
            if feature.startswith("addr_state_"):
                base_features.add("addr_state")
            elif feature.startswith("home_ownership_"):
                base_features.add("home_ownership")
            elif feature.startswith("purpose_"):
                base_features.add("purpose")
            elif feature.startswith("verification_status_"):
                base_features.add("verification_status")
            elif feature.startswith("title_has_"):
                base_features.add("title")
            elif "_" in feature and not feature.replace("_", "").isnumeric():
                parts = feature.split("_")
                if len(parts) > 1:
                    base_features.add(parts[0])
            else:
                base_features.add(feature)

        user_input_features = []
        for feature in base_features:
            if feature not in [
                "purpose_debt_consolidation",
                "verification_status_Verified",
                "verification_status_Source",
                "title_has_car",
                "title_has_medical",
                "title_has_credit",
                "title_has_home",
                "title_has_consolidation",
                "title_has_debt",
                "title_has_card",
            ] and not any(feature + "_" in f for f in self.feature_list):
                user_input_features.append(feature)

        return user_input_features

    def load_artifacts(self):
        """Load model, scaler, and imputer."""
        try:
            model_path = self.model_dir / "xgb_best_model.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            imputer_path = self.model_dir / "imputer.pkl"

            if not model_path.exists():
                raise FileNotFoundError(f"No model file found at {model_path}")

            self.model = joblib.load(model_path)
            print(f"✅ Loaded model: {model_path.name}")

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Loaded scaler: {scaler_path.name}")

            if imputer_path.exists():
                self.imputer = joblib.load(imputer_path)
                print(f"✅ Loaded imputer: {imputer_path.name}")

            if hasattr(self.model, "n_features_in_"):
                print(f"📊 Model expects {self.model.n_features_in_} features")
                print(f"📊 We have {len(self.feature_list)} features in our list")

                if self.model.n_features_in_ != len(self.feature_list):
                    print("⚠️ WARNING: Feature count mismatch!")

        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise

    def _engineer_features(self, df):
        if not self.feature_list:
            raise ValueError("No feature list available!")

        for feature in self.base_features_needed:
            if feature not in df.columns:
                if feature in [
                    "loan_amnt", "annual_inc", "int_rate", "dti", "total_acc",
                    "revol_bal", "total_bc_limit", "total_bal_ex_mort", "avg_cur_bal",
                    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
                    "mths_since_recent_bc", "mths_since_recent_inq", "last_fico_range_low",
                    "last_fico_range_high", "years_since_earliest_cr"
                ]:
                    df[feature] = 0
                elif feature in ["addr_state", "home_ownership", "purpose", "verification_status", "title"]:
                    df[feature] = "unknown"
                elif feature in [
                    "grade_numeric", "emp_length_numeric", "revol_util_decimal",
                    "loan_to_income", "int_rate_times_loan", "subprime_high_dti",
                    "pct_tl_nvr_dlq", "title_length", "title_word_count"
                ]:
                    df[feature] = 0
                elif feature in ["delinq_2yrs", "inq_last_6mths", "open_acc", "has_delinq_history"]:
                    df[feature] = 0
                else:
                    df[feature] = 0

        df = self._create_one_hot_features(df)
        df = self._create_engineered_features(df)
        return df

    def _create_one_hot_features(self, df):
        if not self.feature_list:
            return df

        for feature in self.feature_list:
            if feature.startswith("addr_state_"):
                state_code = feature.replace("addr_state_", "")
                df[feature] = (
                    df["addr_state"].astype(str).str.upper() == state_code
                ).astype(int) if "addr_state" in df.columns else 0

            elif feature.startswith("home_ownership_"):
                ownership_type = feature.replace("home_ownership_", "")
                df[feature] = (
                    df["home_ownership"].astype(str).str.upper() == ownership_type
                ).astype(int) if "home_ownership" in df.columns else 0

            elif feature.startswith("purpose_"):
                purpose_type = feature.replace("purpose_", "")
                df[feature] = (
                    df["purpose"].astype(str).str.lower().str.replace(" ", "_", regex=False) == purpose_type
                ).astype(int) if "purpose" in df.columns else 0

            elif feature.startswith("verification_status_"):
                status_type = feature.replace("verification_status_", "")
                df[feature] = (
                    df["verification_status"].astype(str).str.replace(" ", "_", regex=False) == status_type
                ).astype(int) if "verification_status" in df.columns else 0

            elif feature.startswith("title_has_"):
                keyword = feature.replace("title_has_", "")
                if "title" in df.columns:
                    title_str = str(df["title"].iloc[0]).lower() if len(df) > 0 else ""
                    df[feature] = 1 if keyword in title_str else 0
                else:
                    df[feature] = 0

        return df

    def _create_engineered_features(self, df):
        if "grade" in df.columns:
            grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
            df["grade_numeric"] = df["grade"].map(grade_map).fillna(4)

        if "emp_length" in df.columns:
            df["emp_length_numeric"] = df["emp_length"].apply(self._convert_emp_length)

        if "revol_util" in df.columns:
            df["revol_util_decimal"] = (
                df["revol_util"].astype(str).str.replace("%", "", regex=False).astype(float) / 100
            )

        if "loan_amnt" in df.columns and "annual_inc" in df.columns:
            df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"].replace(0, 1) + 1)

        if "int_rate" in df.columns and "loan_amnt" in df.columns:
            df["int_rate_times_loan"] = df["int_rate"] * df["loan_amnt"] / 1000

        if "delinq_2yrs" in df.columns:
            df["has_delinq_history"] = (df["delinq_2yrs"] > 0).astype(int)

        if "grade_numeric" in df.columns and "dti" in df.columns:
            df["subprime_high_dti"] = ((df["grade_numeric"] >= 4) & (df["dti"] > 20)).astype(int)

        if "title" in df.columns:
            title_str = str(df["title"].iloc[0]).lower() if len(df) > 0 else ""
            df["title_length"] = len(title_str)
            df["title_word_count"] = len(title_str.split())

        if "years_since_earliest_cr" not in df.columns:
            df["years_since_earliest_cr"] = 10

        for feature in self.feature_list:
            if feature not in df.columns and not feature.startswith((
                "addr_state_", "home_ownership_", "purpose_", "verification_status_", "title_has_"
            )):
                if "fico" in feature.lower():
                    df[feature] = 700
                elif any(x in feature for x in ["rate", "util", "pct", "ratio"]):
                    df[feature] = 0.5
                elif any(x in feature for x in ["loan", "amt", "bal", "limit", "inc"]):
                    df[feature] = 0
                elif any(x in feature for x in ["month", "mo", "mth", "year"]):
                    df[feature] = 0
                else:
                    df[feature] = 0

        return df

    def _convert_emp_length(self, val):
        if pd.isna(val):
            return 3.0
        val = str(val).lower()
        if "10+" in val:
            return 10.0
        elif "< 1" in val:
            return 0.5
        else:
            numbers = re.findall(r"\d+", val)
            return float(numbers[0]) if numbers else 3.0

    def preprocess_input(self, input_dict):
        if not self.feature_list:
            raise ValueError("No feature list available!")

        df = pd.DataFrame([input_dict])
        df = self._engineer_features(df)

        processed_df = pd.DataFrame(columns=self.feature_list)
        for feature in self.feature_list:
            if feature in df.columns:
                processed_df[feature] = df[feature].values
            else:
                processed_df[feature] = 0

        print(f"🔧 Created dataframe with {len(processed_df.columns)} features")

        if self.imputer is not None and not processed_df.empty:
            try:
                processed_df = pd.DataFrame(
                    self.imputer.transform(processed_df),
                    columns=self.feature_list
                )
            except Exception as e:
                print(f"⚠️ Imputer error: {e}")

        if self.scaler is not None and not processed_df.empty:
            try:
                processed_df = pd.DataFrame(
                    self.scaler.transform(processed_df),
                    columns=self.feature_list
                )
            except Exception as e:
                print(f"⚠️ Scaler error: {e}")

        return processed_df.values

    def predict(self, input_dict):
        try:
            features = self.preprocess_input(input_dict)

            if features.size == 0:
                raise ValueError("No features generated!")

            print(f"🔧 Processed features shape: {features.shape}")

            default_prob = self.model.predict_proba(features)[0, 1]
            decision = "APPROVE" if default_prob < self.optimal_threshold else "REJECT"

            return {
                "success": True,
                "default_probability": float(default_prob),
                "decision": decision,
                "risk_level": self._get_risk_level(default_prob),
                "confidence": self._get_confidence(default_prob),
                "optimal_threshold": self.optimal_threshold,
                "explanation": f"Default probability: {default_prob:.1%} (threshold: {self.optimal_threshold:.1%})",
            }

        except Exception as e:
            import traceback
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "decision": "ERROR",
            }

    def _get_risk_level(self, prob):
        if prob < 0.2:
            return "LOW"
        elif prob < 0.4:
            return "MEDIUM"
        elif prob < 0.6:
            return "HIGH"
        else:
            return "VERY HIGH"

    def _get_confidence(self, prob):
        distance = abs(prob - self.optimal_threshold)
        return max(0.5, 1.0 - distance * 2)


if __name__ == "__main__":
    print("🧪 Testing CreditRiskPredictor...")
    print("=" * 60)

    predictor = CreditRiskPredictor("model_artifacts")

    if not predictor.feature_list:
        print("\n❌ Cannot proceed without features!")
    else:
        test_loan = {
            "loan_amnt": 15000,
            "int_rate": 12.5,
            "addr_state": "CA",
            "home_ownership": "RENT",
            "purpose": "debt_consolidation",
            "verification_status": "Verified",
            "title": "Debt consolidation loan for credit card payoff",
            "dti": 18.5,
            "annual_inc": 75000,
            "revol_util": "45%",
            "delinq_2yrs": 0,
            "inq_last_6mths": 2,
            "open_acc": 8,
            "total_acc": 25,
            "revol_bal": 5000,
            "total_bc_limit": 20000,
            "total_bal_ex_mort": 30000,
            "avg_cur_bal": 2500,
            "mo_sin_old_il_acct": 60,
            "mo_sin_old_rev_tl_op": 48,
            "mo_sin_rcnt_rev_tl_op": 12,
            "mths_since_recent_bc": 6,
            "mths_since_recent_inq": 3,
            "pct_tl_nvr_dlq": 0.95,
            "last_fico_range_low": 680,
            "last_fico_range_high": 684,
            "grade": "C",
            "emp_length": "5 years",
            "years_since_earliest_cr": 10,
        }

        print(f"\n📊 Making test prediction...")
        print(f"Using input with {len(test_loan)} fields")

        result = predictor.predict(test_loan)

        print("\n" + "=" * 60)
        print("📈 PREDICTION RESULTS:")
        print("=" * 60)
        for key, value in result.items():
            if key != "explanation" or result["success"]:
                print(f"{key:25}: {value}")