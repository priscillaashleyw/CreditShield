import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from load_data import DataLoader
from train_tab_transformer import TabTransformerTrainer


def load_config():
    config_path = Path(__file__).parent.parent / 'config' / 'tab_transformer_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_for_local_data():
    data_dir = Path(__file__).parent / 'data'
    if not data_dir.exists():
        return None

    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        return None

    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(csvs[0])


def get_probabilities(trainer, model, X):
    # adjust this part if your trainer exposes probabilities differently

    if hasattr(trainer, "predict_proba"):
        probs = trainer.predict_proba(model, X)
        return np.asarray(probs).reshape(-1)

    if hasattr(trainer, "predict"):
        out = trainer.predict(model, X)

        if isinstance(out, dict):
            if "y_prob" in out:
                return np.asarray(out["y_prob"]).reshape(-1)
            if "probabilities" in out:
                return np.asarray(out["probabilities"]).reshape(-1)

        if isinstance(out, tuple) and len(out) >= 2:
            return np.asarray(out[1]).reshape(-1)

        arr = np.asarray(out).reshape(-1)
        if np.all((arr >= 0) & (arr <= 1)):
            return arr

    raise RuntimeError("Could not extract probabilities from trainer/model.")


def main():
    print("=" * 70)
    print("RANDOM SAMPLE PROBABILITY TEST")
    print("=" * 70)

    local_data = check_for_local_data()
    if not local_data:
        print("No local CSV found in training/data/")
        sys.exit(1)

    config = load_config()
    config["paths"]["raw_data"] = local_data

    # same style as train_model.py
    training_config = config.get("training", {})
    config.update(training_config)

    loader = DataLoader(config)
    df = loader.load_and_filter_data()
    df_clean = loader.define_target(df, strategy="business")

    X_train, X_test, y_train, y_test = loader.random_split(
        df_clean,
        test_size=0.2,
        random_state=42
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = TabTransformerTrainer(config, device=device)

    print("\nTraining model...")
    model, metrics = trainer.train(X_train, X_test, y_train, y_test)

    print("\nFinal test metrics:")
    for k, v in metrics.items():
        try:
            print(f"{k}: {float(v):.4f}")
        except:
            print(f"{k}: {v}")

    # randomly sample rows from the Lending Club TEST set
    n_samples = min(5, len(X_test))
    sampled_idx = X_test.sample(n=n_samples, random_state=42).index

    sample_X = X_test.loc[sampled_idx].copy()
    sample_y = y_test.loc[sampled_idx].copy()

    sample_probs = get_probabilities(trainer, model, sample_X)
    sample_pred = (sample_probs >= 0.287).astype(int)

    out = sample_X.copy()
    out["actual_default"] = sample_y.values
    out["predicted_probability_default"] = sample_probs
    out["predicted_class_at_0.287"] = sample_pred

    cols_to_show = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "term",
        "actual_default",
        "predicted_probability_default",
        "predicted_class_at_0.287",
    ]
    cols_to_show = [c for c in cols_to_show if c in out.columns]

    print("\n" + "=" * 70)
    print("RANDOMLY SAMPLED TEST INSTANCES")
    print("=" * 70)
    print(out[cols_to_show].to_string(index=True))

    print("\n" + "=" * 70)
    print("ONE INSTANCE EXAMPLE")
    print("=" * 70)
    one = out.iloc[0]
    for col in cols_to_show:
        val = one[col]
        if col == "predicted_probability_default":
            print(f"{col}: {val:.4f}")
        else:
            print(f"{col}: {val}")


if __name__ == "__main__":
    main()