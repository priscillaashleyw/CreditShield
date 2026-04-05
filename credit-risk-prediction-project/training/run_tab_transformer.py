#!/usr/bin/env python3
"""
run_tab_transformer.py
======================
Entry point for the TabTransformer credit-risk training pipeline.

Run from the training/ directory:
    python run_tab_transformer.py

Pipeline stages
---------------
1. Load config  → config/tab_transformer_config.yaml
2. Load data    → src/load_data.DataLoader
                  (filters to 2013-2014, removes leakage cols,
                   handles structural missingness indicators)
3. Define target → binary: 1 = default, 0 = good
4. Split         → stratified random 80 / 20 (no time leakage)
5. Train         → src/train_tab_transformer.TabTransformerTrainer
                  (prepare → build → train loop → evaluate)
6. Persist       → results/best_tab_transformer.pth
                   results/metrics.yaml
"""

import sys
from pathlib import Path

# ── Make src/ importable so that train_tab_transformer can do
#    `from tab_transformer import TabTransformer` without a package install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
from src.load_data import DataLoader
from src.train_tab_transformer import TabTransformerTrainer


def main():
    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    config_path = Path(__file__).parent.parent / "config" / "tab_transformer_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Expected location: credit-risk-prediction-project/config/"
            "tab_transformer_config.yaml"
        )

    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("TabTransformer  │  Credit Risk Prediction")
    print("Output: P(default)  →  probability in [0, 1]")
    cat_features = config.get('categorical_features', [])
    num_features = config.get('numerical_features',   [])
    print(f"Config: {len(cat_features)} categorical, "
          f"{len(num_features)} numerical features defined")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 2. Load & filter raw data  (via load_data.DataLoader)
    # ------------------------------------------------------------------
    print("\n[1/3] Loading data ...")
    data_loader = DataLoader(config)

    # load_and_filter_data():
    #   - Loads essential_columns from the Kaggle CSV
    #   - Filters to years 2013-2014
    #   - Drops post-origination leakage columns
    #   - Applies log1p transform to annual_inc and revol_bal
    #   - Creates structural-missingness indicator columns
    #     (mths_since_last_*_missing) for bureau-event columns
    df = data_loader.load_and_filter_data()

    # ------------------------------------------------------------------
    # 3. Define binary target
    #    1 = default (Charged Off / Default / Late 31-120)
    #    0 = good    (Fully Paid)
    # ------------------------------------------------------------------
    df = data_loader.define_target(df, strategy='business')

    # ------------------------------------------------------------------
    # 4. Stratified random train / test split  (80 / 20)
    #    Returns: X_train, X_test (DataFrames), y_train, y_test (Series)
    #    Note: loan_status and target columns are removed from X.
    #          Date columns (issue_d, issue_date, issue_year) remain in X
    #          and are dropped inside TabTransformerTrainer.prepare_data().
    # ------------------------------------------------------------------
    print("\n[2/3] Splitting data ...")
    X_train, X_test, y_train, y_test = data_loader.random_split(df, test_size=0.2)

    # ------------------------------------------------------------------
    # 5. Train and evaluate TabTransformer
    # ------------------------------------------------------------------
    print("\n[3/3] Training TabTransformer ...")
    trainer = TabTransformerTrainer(config)
    model, metrics = trainer.train(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------
    # 6. Persist results
    # ------------------------------------------------------------------
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics.yaml"
    metrics_to_save = {
        'model_type':        'TabTransformer',
        'output_type':       'probability (0=good, 1=default)',
        'optimal_threshold': float(metrics['threshold']),
        'auc':               float(metrics['auc']),
        'accuracy':          float(metrics['accuracy']),
        'precision':         float(metrics['precision']),
        'recall':            float(metrics['recall']),
        'f1':                float(metrics['f1']),
        'mae':               float(metrics['mae']),
        'true_positives':    int(metrics['true_positives']),
        'true_negatives':    int(metrics['true_negatives']),
        'false_positives':   int(metrics['false_positives']),
        'false_negatives':   int(metrics['false_negatives']),
    }

    with open(metrics_path, 'w') as f:
        yaml.dump(metrics_to_save, f, default_flow_style=False, sort_keys=False)

    model_path = results_dir / 'best_tab_transformer.pth'
    print(f"\n✅  Training complete!")
    print(f"   Model checkpoint → {model_path}")
    print(f"   Metrics YAML     → {metrics_path}")

    return metrics


if __name__ == "__main__":
    main()
