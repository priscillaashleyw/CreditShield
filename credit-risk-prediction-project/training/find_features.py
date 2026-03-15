# training/find_features.py
import json
import joblib
from pathlib import Path


def _pick_latest(files):
    files = list(files)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def analyze_training_features():
    print("🔍 Analyzing training features...")
    print("=" * 60)

    training_dir = Path(__file__).resolve().parent
    project_root = training_dir.parent
    models_dir = training_dir / "models"
    deployment_artifacts_dir = project_root / "deployment" / "model_artifacts"
    deployment_artifacts_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = _pick_latest(models_dir.glob("scaler_our_enhanced_*.pkl"))
    if scaler_path is None:
        scaler_path = _pick_latest(models_dir.glob("scaler_paper_16_*.pkl"))

    imputer_path = _pick_latest(models_dir.glob("imputer_our_enhanced_*.pkl"))
    if imputer_path is None:
        imputer_path = _pick_latest(models_dir.glob("imputer_paper_16_*.pkl"))

    if scaler_path is None and imputer_path is None:
        raise FileNotFoundError("No scaler/imputer artifacts found in training/models")

    feature_names = None
    source = None

    if scaler_path is not None:
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, "feature_names_in_"):
            feature_names = list(scaler.feature_names_in_)
            source = f"scaler: {scaler_path.name}"

    if feature_names is None and imputer_path is not None:
        imputer = joblib.load(imputer_path)
        if hasattr(imputer, "feature_names_in_"):
            feature_names = list(imputer.feature_names_in_)
            source = f"imputer: {imputer_path.name}"

    if feature_names is None:
        raise RuntimeError(
            "Could not recover exact training feature names from scaler/imputer. "
            "Artifacts do not expose feature_names_in_."
        )

    print(f"✅ Recovered exact feature list from {source}")
    print(f"✅ Feature count: {len(feature_names)}")

    json_path = deployment_artifacts_dir / "training_features.json"
    csv_path = deployment_artifacts_dir / "training_features.csv"

    payload = {
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "source": source,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w") as f:
        f.write("feature_name\n")
        for name in feature_names:
            f.write(f"{name}\n")

    print(f"✅ Saved feature JSON to: {json_path}")
    print(f"✅ Saved feature CSV to:  {csv_path}")

    return feature_names


if __name__ == "__main__":
    analyze_training_features()