"""
train_tab_transformer.py
========================
Training and evaluation pipeline for the TabTransformer credit-risk model.

Integrates with load_data.DataLoader:
  - Call prepare_data() with the DataFrames from DataLoader.random_split()
  - Call train() to run the full pipeline

Pipeline design
---------------
  Categorical features → LabelEncoder → Embedding + Transformer layers
  Numerical features   → median imputation + StandardScaler → bypass path
  Both streams         → concatenated → MLP → sigmoid → P(default)

Loss
----
  Weighted BCELoss (model already outputs sigmoid probabilities).
  Positive-class weight = n_neg / n_pos to compensate for class imbalance.

Early stopping
--------------
  Tracks AUC + F1 combined score; saves best checkpoint automatically.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from tab_transformer import TabTransformer


class TabTransformerTrainer:
    """
    Encapsulates the full train / evaluate pipeline for TabTransformer.

    Parameters
    ----------
    config : dict
        Loaded from tab_transformer_config.yaml.  Must contain keys:
        paths, data_settings, model, batch_size, epochs, learning_rate,
        weight_decay, early_stopping_patience.
        Optional: categorical_features, numerical_features (explicit lists).
    device : str
        'cuda' or 'cpu'.  Auto-detected if not provided.
    """

    def __init__(self, config, device=None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Feature lists ──────────────────────────────────────────────────
        # Populated from config when explicit; otherwise auto-detected by dtype.
        self._config_categorical: list = config.get('categorical_features', []) or []
        self._config_numerical:   list = config.get('numerical_features',   []) or []

        self.categorical_features: list = []
        self.numerical_features:   list = []
        self.categorical_dims:     list = []   # vocabulary size per categorical col

        # ── Fitted preprocessing objects ───────────────────────────────────
        self.label_encoders: dict = {}         # col → LabelEncoder (fit on train)
        self.scaler: StandardScaler = None     # fit on train numerical features

        # ── Threshold used when converting probabilities to class predictions
        self.best_threshold: float = 0.5

        # ── DataLoaders (populated by prepare_data) ────────────────────────
        self.train_loader = None
        self.test_loader  = None

        # ── Positive-class weight for imbalanced BCE loss ──────────────────
        self.pos_weight = None

        # Output directory for checkpoints and history
        self.results_dir = Path(__file__).parent.parent / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Data preparation
    # ======================================================================

    def prepare_data(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: pd.Series,
        y_test:  pd.Series,
    ):
        """
        Prepare train and test splits for TabTransformer.

        Steps
        -----
        1. Drop metadata columns not useful for modelling (issue dates).
        2. Determine numerical vs categorical feature lists:
           - Use explicit config lists filtered to columns that actually exist.
           - Auto-add any unlisted numeric columns (e.g. dynamic _missing
             indicator columns added by load_data.py).
           - Fall back to dtype auto-detection when config lists are empty.
        3. Numerical:  coerce to float → median imputation → StandardScaler.
        4. Categorical: fillna('NA') → LabelEncoder (train); unseen test
           categories are mapped to index 0 (a safe fallback).
        5. Compute positive-class weight for imbalanced loss.
        6. Build PyTorch TensorDatasets and DataLoaders.

        Returns
        -------
        (train_loader, test_loader) – both are torch.utils.data.DataLoader
        """
        print("\nPreparing data for TabTransformer...")

        # ── 1. Drop metadata columns ───────────────────────────────────────
        meta_cols = ['issue_d', 'issue_date', 'issue_year']
        X_train = X_train.drop(columns=[c for c in meta_cols if c in X_train.columns],
                                errors='ignore')
        X_test  = X_test.drop(columns=[c for c in meta_cols if c in X_test.columns],
                               errors='ignore')

        # ── 2a. Categorical feature list ──────────────────────────────────
        if self._config_categorical:
            # Filter to columns that actually arrived in the DataFrame
            self.categorical_features = [
                c for c in self._config_categorical if c in X_train.columns
            ]
            missing = [c for c in self._config_categorical if c not in X_train.columns]
            if missing:
                print(f"  [WARN] Categorical cols in config but absent in data: {missing}")
        else:
            # Fallback: auto-detect by dtype
            self.categorical_features = (
                X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            )

        # ── 2b. Numerical feature list ────────────────────────────────────
        known_cols = set(self.categorical_features) | set(meta_cols)

        if self._config_numerical:
            self.numerical_features = [
                c for c in self._config_numerical if c in X_train.columns
            ]
            missing_num = [c for c in self._config_numerical if c not in X_train.columns]
            if missing_num:
                print(f"  [WARN] Numerical cols in config but absent in data: {missing_num}")

            # Auto-capture any numeric columns NOT in either explicit list.
            # This picks up dynamic columns added by load_data.py at runtime
            # (e.g. mths_since_last_delinq_missing).
            listed = set(self.numerical_features) | known_cols
            extra_num = [
                c for c in X_train.columns
                if c not in listed
                and pd.api.types.is_numeric_dtype(X_train[c])
            ]
            if extra_num:
                print(f"  Auto-adding {len(extra_num)} unlisted numeric col(s): {extra_num}")
                self.numerical_features.extend(extra_num)
        else:
            # Fallback: auto-detect by dtype, exclude categorical columns
            self.numerical_features = [
                c for c in X_train.select_dtypes(
                    include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8']
                ).columns
                if c not in known_cols
            ]

        print(f"  Numerical features  : {len(self.numerical_features)}")
        print(f"  Categorical features: {len(self.categorical_features)}")

        if not self.categorical_features:
            raise ValueError(
                "TabTransformer requires at least 1 categorical feature.\n"
                "Set 'categorical_features' in tab_transformer_config.yaml "
                "or ensure the data contains string/category columns."
            )

        # ── 3. Numerical: impute then scale ───────────────────────────────
        X_train_num = X_train[self.numerical_features].copy()
        X_test_num  = X_test[self.numerical_features].copy()

        # Coerce to float (handles edge cases where a numeric col is stored
        # as object with stray string values)
        for col in self.numerical_features:
            X_train_num[col] = pd.to_numeric(X_train_num[col], errors='coerce')
            X_test_num[col]  = pd.to_numeric(X_test_num[col],  errors='coerce')

        # Impute NaN with train-set column median (no test leakage)
        for col in self.numerical_features:
            median_val = X_train_num[col].median()
            X_train_num[col] = X_train_num[col].fillna(median_val)
            X_test_num[col]  = X_test_num[col].fillna(median_val)

        self.scaler = StandardScaler()
        X_train_num_scaled = self.scaler.fit_transform(X_train_num)   # fit on train
        X_test_num_scaled  = self.scaler.transform(X_test_num)        # transform only

        # ── 4. Categorical: label-encode each column ──────────────────────
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_test_cat  = pd.DataFrame(index=X_test.index)
        self.categorical_dims = []

        for col in self.categorical_features:
            le = LabelEncoder()
            # Use 'NA' as the explicit missing category
            train_vals = X_train[col].fillna('NA').astype(str)
            le.fit(train_vals)
            X_train_cat[col] = le.transform(train_vals)

            # Map unseen test categories to index 0 (safe fallback)
            test_vals = X_test[col].fillna('NA').astype(str)
            X_test_cat[col] = test_vals.apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

            self.label_encoders[col] = le
            self.categorical_dims.append(len(le.classes_))
            print(f"    {col}: {len(le.classes_)} categories")

        # ── 5. Positive-class weight for imbalanced BCE ───────────────────
        n_samples = len(y_train)
        n_pos     = int(y_train.sum())
        n_neg     = n_samples - n_pos
        self.pos_weight = (
            torch.FloatTensor([n_neg / n_pos]).to(self.device)
            if n_pos > 0 else torch.FloatTensor([1.0]).to(self.device)
        )
        print(f"\n  Class imbalance ratio  : {n_neg / n_pos:.2f}:1 (neg:pos)")
        print(f"  Positive-class weight  : {self.pos_weight.item():.2f}")

        # ── 6. Build PyTorch tensors and DataLoaders ──────────────────────
        X_train_num_t = torch.FloatTensor(X_train_num_scaled).to(self.device)
        X_test_num_t  = torch.FloatTensor(X_test_num_scaled).to(self.device)
        X_train_cat_t = torch.LongTensor(X_train_cat.values).to(self.device)
        X_test_cat_t  = torch.LongTensor(X_test_cat.values).to(self.device)
        # Float targets required by BCELoss
        y_train_t = torch.FloatTensor(y_train.values).to(self.device)
        y_test_t  = torch.FloatTensor(y_test.values).to(self.device)

        batch_size = int(self.config.get('batch_size', 256))
        train_dataset = TensorDataset(X_train_num_t, X_train_cat_t, y_train_t)
        test_dataset  = TensorDataset(X_test_num_t,  X_test_cat_t,  y_test_t)

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self.test_loader  = DataLoader(
            test_dataset,  batch_size=batch_size, shuffle=False
        )

        print(f"  Train batches: {len(self.train_loader)}, "
              f"Test batches: {len(self.test_loader)}")
        return self.train_loader, self.test_loader

    # ======================================================================
    # Model construction
    # ======================================================================

    def build_model(self) -> TabTransformer:
        """
        Instantiate TabTransformer from config hyperparameters.

        Clamps attention heads to never exceed num_categorical_features
        (the transformer sequence length).
        """
        model_cfg = self.config.get('model', {})

        # heads must be <= sequence length (num_categorical_features)
        heads = min(
            int(model_cfg.get('heads', 4)),
            len(self.categorical_features)
        )

        model = TabTransformer(
            num_numerical_features  = len(self.numerical_features),
            num_categorical_features= len(self.categorical_features),
            categorical_dims        = self.categorical_dims,
            embedding_dim           = int(model_cfg.get('embedding_dim', 32)),
            depth                   = int(model_cfg.get('depth', 4)),
            heads                   = heads,
            dim_head                = int(model_cfg.get('dim_head', 32)),
            mlp_dim                 = int(model_cfg.get('mlp_dim', 256)),
            num_classes             = 1,          # single sigmoid output
            dropout                 = float(model_cfg.get('dropout', 0.2)),
        ).to(self.device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  TabTransformer built  : {n_params:,} parameters")
        print(f"    Categorical tokens  : {len(self.categorical_features)} "
              f"(embed_dim={model_cfg.get('embedding_dim', 32)})")
        print(f"    Numerical bypass    : {len(self.numerical_features)} features")
        print(f"    Attention heads     : {heads}, depth={model_cfg.get('depth', 4)}, "
              f"dim_head={model_cfg.get('dim_head', 32)}")
        return model

    # ======================================================================
    # Loss function
    # ======================================================================

    def _weighted_bce_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Weighted Binary Cross-Entropy loss for imbalanced classes.

        Because the model already applies sigmoid, we use BCELoss (not
        BCEWithLogitsLoss).  Positive-class samples are up-weighted by
        pos_weight = n_neg / n_pos to compensate for class imbalance.

        Args:
            probs   : (batch,) predicted probabilities in [0, 1]
            targets : (batch,) binary float labels {0.0, 1.0}
        Returns:
            Scalar weighted mean loss.
        """
        bce = nn.functional.binary_cross_entropy(probs, targets, reduction='none')
        weights = torch.where(
            targets == 1,
            self.pos_weight.expand_as(targets),
            torch.ones_like(targets),
        )
        return (bce * weights).mean()

    # ======================================================================
    # Training
    # ======================================================================

    def train_epoch(self, model: TabTransformer, optimizer: optim.Optimizer) -> float:
        """
        Run one full training epoch.

        Returns the mean training loss over all batches.
        """
        model.train()
        total_loss = 0.0

        for X_num, X_cat, y in self.train_loader:
            optimizer.zero_grad()
            probs = model(X_num, X_cat)           # → (batch,) in [0, 1]
            loss  = self._weighted_bce_loss(probs, y)
            loss.backward()
            # Gradient clipping prevents exploding gradients in deep transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ======================================================================
    # Evaluation
    # ======================================================================

    def find_optimal_threshold(self, labels: np.ndarray, probs: np.ndarray) -> float:
        """
        Find the decision threshold that maximises F1 on the
        precision-recall curve.
        """
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx  = np.argmax(f1_scores)
        return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    def evaluate(self, model: TabTransformer, find_threshold: bool = False):
        """
        Evaluate the model on the held-out test set.

        Args:
            model          : Trained TabTransformer.
            find_threshold : If True, re-tune the decision threshold via
                             the precision-recall curve (use every 5 epochs
                             during training, and once on the final model).

        Returns:
            metrics   : dict with AUC, accuracy, precision, recall, F1,
                        MAE, optimal threshold, and confusion-matrix counts.
            all_probs : np.ndarray of predicted default probabilities.
        """
        model.eval()
        all_probs:  list = []
        all_labels: list = []

        with torch.no_grad():
            for X_num, X_cat, y in self.test_loader:
                probs = model(X_num, X_cat)   # already (0-1) via sigmoid
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Optionally re-calibrate decision threshold
        if find_threshold:
            self.best_threshold = self.find_optimal_threshold(all_labels, all_probs)

        all_preds = (all_probs >= self.best_threshold).astype(int)

        # ── Classification metrics ────────────────────────────────────────
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5   # undefined when only one class present

        mae = mean_absolute_error(all_labels, all_probs)
        cm  = confusion_matrix(all_labels.astype(int), all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            'auc':             auc,
            'accuracy':        (tp + tn) / (tp + tn + fp + fn),
            'precision':       precision,
            'recall':          recall,
            'f1':              f1,
            'mae':             mae,
            'threshold':       self.best_threshold,
            'true_positives':  int(tp),
            'true_negatives':  int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }, all_probs

    # ======================================================================
    # Full training pipeline
    # ======================================================================

    def train(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: pd.Series,
        y_test:  pd.Series,
    ):
        """
        Full pipeline: prepare → build → train loop → evaluate.

        Training details
        ----------------
        - Optimiser  : AdamW (weight decay for regularisation)
        - Scheduler  : CosineAnnealingWarmRestarts (smooth LR decay + warm restarts)
        - Loss       : Weighted BCELoss (pos_weight for class imbalance)
        - Stopping   : Early stopping on AUC + F1; best checkpoint is reloaded
        - Threshold  : Decision threshold re-tuned every 5 epochs and at end

        Args:
            X_train, X_test : DataFrames from DataLoader.random_split()
            y_train, y_test : binary Series (1 = default, 0 = good)

        Returns:
            model   : best TabTransformer loaded from checkpoint
            metrics : dict of final test-set metrics
        """
        # ── Stage 1: data preparation ─────────────────────────────────────
        self.prepare_data(X_train, X_test, y_train, y_test)

        # ── Stage 2: model construction ───────────────────────────────────
        model = self.build_model()

        # ── Stage 3: training configuration ──────────────────────────────
        lr           = float(self.config.get('learning_rate', 0.0005))
        weight_decay = float(self.config.get('weight_decay', 0.01))
        epochs       = int(self.config.get('epochs', 100))
        patience     = int(self.config.get('early_stopping_patience', 15))

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Cosine annealing with warm restarts: T_0=10 epochs first cycle,
        # doubles each subsequent cycle (T_mult=2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        best_score   = 0.0
        patience_ctr = 0
        checkpoint   = self.results_dir / 'best_tab_transformer.pth'
        history      = {'train_loss': [], 'val_auc': [], 'val_f1': [], 'val_recall': []}

        print(f"\nStarting training for up to {epochs} epochs "
              f"(patience={patience}) ...")
        print(f"  LR={lr}, weight_decay={weight_decay}, "
              f"batch_size={self.config.get('batch_size', 256)}, "
              f"device={self.device}")
        print("-" * 65)

        # ── Stage 4: training loop ────────────────────────────────────────
        for epoch in range(epochs):
            train_loss = self.train_epoch(model, optimizer)
            scheduler.step()

            # Re-tune decision threshold every 5 epochs
            find_thresh = (epoch % 5 == 0)
            metrics, _  = self.evaluate(model, find_threshold=find_thresh)

            history['train_loss'].append(train_loss)
            history['val_auc'].append(metrics['auc'])
            history['val_f1'].append(metrics['f1'])
            history['val_recall'].append(metrics['recall'])

            cur_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1:3d}/{epochs} │ "
                  f"Loss: {train_loss:.4f} │ LR: {cur_lr:.2e}")
            print(f"  AUC: {metrics['auc']:.4f} │ "
                  f"F1: {metrics['f1']:.4f} │ "
                  f"Prec: {metrics['precision']:.4f} │ "
                  f"Rec: {metrics['recall']:.4f}")
            print(f"  Threshold: {metrics['threshold']:.3f} │ "
                  f"TP: {metrics['true_positives']:,} │ "
                  f"FP: {metrics['false_positives']:,}")

            # Combined score for early stopping (penalises poor recall)
            score = metrics['auc'] + metrics['f1']
            if score > best_score:
                best_score   = score
                patience_ctr = 0
                torch.save(model.state_dict(), checkpoint)
                print(f"  ✓ Best model saved (AUC+F1={score:.4f})")
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                print(f"\n⚠  Early stopping after epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                break

        # ── Stage 5: reload best checkpoint & final evaluation ────────────
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        final_metrics, _ = self.evaluate(model, find_threshold=True)

        # ── Stage 6: print final report ───────────────────────────────────
        print("\n" + "=" * 65)
        print("FINAL TEST-SET RESULTS")
        print("=" * 65)
        print(f"  Optimal threshold : {final_metrics['threshold']:.4f}")
        print(f"  AUC-ROC           : {final_metrics['auc']:.4f}")
        print(f"  Accuracy          : {final_metrics['accuracy']:.4f}")
        print(f"  Precision         : {final_metrics['precision']:.4f}")
        print(f"  Recall            : {final_metrics['recall']:.4f}")
        print(f"  F1-Score          : {final_metrics['f1']:.4f}")
        print(f"  MAE               : {final_metrics['mae']:.4f}")
        print("-" * 65)
        print("  Confusion Matrix  (rows=actual, cols=predicted):")
        print(f"    TP: {final_metrics['true_positives']:>7,}  │  "
              f"FP: {final_metrics['false_positives']:>7,}")
        print(f"    FN: {final_metrics['false_negatives']:>7,}  │  "
              f"TN: {final_metrics['true_negatives']:>7,}")
        print("=" * 65)

        # ── Stage 7: persist training history ─────────────────────────────
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_path = self.results_dir / f'history_{ts}.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n  Training history saved → {history_path}")

        return model, final_metrics
