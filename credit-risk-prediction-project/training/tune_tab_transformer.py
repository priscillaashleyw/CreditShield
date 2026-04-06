#!/usr/bin/env python3
"""
tune_tab_transformer.py
=======================
Bayesian hyperparameter optimisation for the TabTransformer using Optuna.

Uses a hold-out validation set (carved from the training split) to avoid
test-set leakage during tuning.  After tuning, retrains the best config
on the full training set and evaluates on the held-out test set.

Run from the training/ directory:
    python tune_tab_transformer.py               # 30 trials (default)
    python tune_tab_transformer.py --n-trials 50  # custom trial count

Outputs:
    results/tuning_results.yaml     – best hyperparameters + final metrics
    results/tuning_study.csv        – all trial results for analysis
    results/best_tab_transformer.pth – retrained best model checkpoint
    results/metrics.yaml            – final test-set metrics
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import optuna
import torch
import yaml
from sklearn.model_selection import train_test_split

from src.load_data import DataLoader
from src.train_tab_transformer import TabTransformerTrainer


# =====================================================================
# Optuna objective: one trial = one full train + evaluate cycle
# =====================================================================

class TabTransformerObjective:
    """
    Optuna objective that trains a TabTransformer with sampled
    hyperparameters and returns a metric to maximise.

    We split the original training set into train/val (85/15) for tuning
    so the test set is never touched until final evaluation.
    """

    def __init__(self, config, X_train, y_train, device):
        self.base_config = config
        self.device = device

        # Create train/val split for tuning  (stratified, reproducible)
        self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            random_state=42,
            stratify=y_train,
        )
        print(f"\nTuning splits: train={len(self.X_tr):,}, "
              f"val={len(self.X_val):,}")

    def __call__(self, trial: optuna.Trial) -> float:
        """Sample hyperparameters, train, return AUC + F1."""

        # ── Sample hyperparameters ────────────────────────────────────
        # Architecture
        embedding_dim = trial.suggest_categorical(
            'embedding_dim', [16, 32, 64]
        )
        depth = trial.suggest_int('depth', 2, 6)
        heads = trial.suggest_categorical('heads', [1, 2, 4])
        dim_head = trial.suggest_categorical('dim_head', [16, 32, 64])
        mlp_dim = trial.suggest_categorical('mlp_dim', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.05, 0.4, step=0.05)

        # Training
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        # Clamp heads to categorical feature count
        n_cat = len(self.base_config.get('categorical_features', []) or [])
        if n_cat > 0:
            heads = min(heads, n_cat)

        # ── Build trial config ────────────────────────────────────────
        trial_config = {
            **self.base_config,
            'model': {
                'embedding_dim': embedding_dim,
                'depth': depth,
                'heads': heads,
                'dim_head': dim_head,
                'mlp_dim': mlp_dim,
                'dropout': dropout,
            },
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'epochs': 40,                  # shorter for tuning speed
            'early_stopping_patience': 8,  # faster early stopping
        }

        # ── Train and evaluate ────────────────────────────────────────
        try:
            trainer = TabTransformerTrainer(trial_config, device=self.device)
            # Suppress verbose output during tuning
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                model, metrics = trainer.train(
                    self.X_tr.copy(), self.X_val.copy(),
                    self.y_tr.copy(), self.y_val.copy(),
                )

            score = metrics['auc'] + metrics['f1']

            # Report intermediate metrics for pruning
            trial.set_user_attr('auc', metrics['auc'])
            trial.set_user_attr('f1', metrics['f1'])
            trial.set_user_attr('precision', metrics['precision'])
            trial.set_user_attr('recall', metrics['recall'])
            trial.set_user_attr('accuracy', metrics['accuracy'])
            trial.set_user_attr('threshold', metrics['threshold'])

            print(f"  Trial {trial.number:3d} │ "
                  f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
                  f"│ score={score:.4f}")

            return score

        except Exception as e:
            print(f"  Trial {trial.number:3d} │ FAILED: {e}")
            return 0.0


# =====================================================================
# Main: run Optuna study, then retrain best on full train set
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Tune TabTransformer hyperparameters")
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of Optuna trials (default: 30)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu (auto-detected)')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # 1. Load config and data
    # ------------------------------------------------------------------
    config_path = Path(__file__).parent.parent / "config" / "tab_transformer_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 65)
    print("TabTransformer  │  Bayesian Hyperparameter Tuning (Optuna)")
    print(f"Trials: {args.n_trials}  │  Device: {device}")
    print("=" * 65)

    print("\n[1/4] Loading data ...")
    data_loader = DataLoader(config)
    df = data_loader.load_and_filter_data()
    df = data_loader.define_target(df, strategy='business')
    X_train, X_test, y_train, y_test = data_loader.random_split(df, test_size=0.2)

    # ------------------------------------------------------------------
    # 2. Run Optuna study
    # ------------------------------------------------------------------
    print("\n[2/4] Starting Optuna study ...")
    print("-" * 65)

    objective = TabTransformerObjective(config, X_train, y_train, device)

    # TPE sampler for Bayesian optimisation; MedianPruner for early stopping
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='tabtransformer_tuning',
    )

    # Suppress Optuna's own logging (our objective prints progress)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # ------------------------------------------------------------------
    # 3. Report best trial
    # ------------------------------------------------------------------
    best = study.best_trial
    print("\n" + "=" * 65)
    print("BEST TRIAL")
    print("=" * 65)
    print(f"  Trial #{best.number}")
    print(f"  Score (AUC+F1)  : {best.value:.4f}")
    print(f"  AUC             : {best.user_attrs.get('auc', 'N/A')}")
    print(f"  F1              : {best.user_attrs.get('f1', 'N/A')}")
    print(f"  Precision       : {best.user_attrs.get('precision', 'N/A')}")
    print(f"  Recall          : {best.user_attrs.get('recall', 'N/A')}")
    print(f"\n  Hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:20s}: {v}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 4. Retrain best config on full training set, evaluate on test set
    # ------------------------------------------------------------------
    print("\n[3/4] Retraining best config on full training set ...")

    best_config = {
        **config,
        'model': {
            'embedding_dim': best.params['embedding_dim'],
            'depth':         best.params['depth'],
            'heads':         best.params['heads'],
            'dim_head':      best.params['dim_head'],
            'mlp_dim':       best.params['mlp_dim'],
            'dropout':       best.params['dropout'],
        },
        'learning_rate':          best.params['learning_rate'],
        'weight_decay':           best.params['weight_decay'],
        'batch_size':             best.params['batch_size'],
        'epochs':                 100,   # full training budget for final run
        'early_stopping_patience': 15,
    }

    trainer = TabTransformerTrainer(best_config, device=device)
    model, final_metrics = trainer.train(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------
    # 5. Save everything
    # ------------------------------------------------------------------
    print("\n[4/4] Saving results ...")
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save final metrics
    metrics_to_save = {
        'model_type':        'TabTransformer (tuned)',
        'output_type':       'probability (0=good, 1=default)',
        'optimal_threshold': float(final_metrics['threshold']),
        'auc':               float(final_metrics['auc']),
        'accuracy':          float(final_metrics['accuracy']),
        'precision':         float(final_metrics['precision']),
        'recall':            float(final_metrics['recall']),
        'f1':                float(final_metrics['f1']),
        'mae':               float(final_metrics['mae']),
        'true_positives':    int(final_metrics['true_positives']),
        'true_negatives':    int(final_metrics['true_negatives']),
        'false_positives':   int(final_metrics['false_positives']),
        'false_negatives':   int(final_metrics['false_negatives']),
    }
    with open(results_dir / 'metrics.yaml', 'w') as f:
        yaml.dump(metrics_to_save, f, default_flow_style=False, sort_keys=False)

    # Save tuning results (best params + study summary)
    tuning_results = {
        'best_trial':       best.number,
        'best_score':       float(best.value),
        'best_params':      best.params,
        'best_val_metrics': {
            k: float(v) for k, v in best.user_attrs.items()
        },
        'final_test_metrics': {
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in final_metrics.items()
        },
        'n_trials':          args.n_trials,
        'timestamp':         datetime.now().isoformat(),
    }
    with open(results_dir / 'tuning_results.yaml', 'w') as f:
        yaml.dump(tuning_results, f, default_flow_style=False, sort_keys=False)

    # Save full study as CSV for analysis
    study_df = study.trials_dataframe()
    study_df.to_csv(results_dir / 'tuning_study.csv', index=False)

    print(f"\n✅  Tuning complete!")
    print(f"   Best params      → {results_dir / 'tuning_results.yaml'}")
    print(f"   Study details    → {results_dir / 'tuning_study.csv'}")
    print(f"   Model checkpoint → {results_dir / 'best_tab_transformer.pth'}")
    print(f"   Final metrics    → {results_dir / 'metrics.yaml'}")


if __name__ == "__main__":
    main()
