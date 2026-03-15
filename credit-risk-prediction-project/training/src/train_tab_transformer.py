"""
TabTransformer Trainer - Training loop for credit risk prediction
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_absolute_error, precision_recall_curve
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from tab_transformer import TabTransformer

class TabTransformerTrainer:
    """Trainer for TabTransformer on credit risk data"""
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.config = config
        self.device = device
        self.numerical_features = config.get('numerical_features', [])
        self.categorical_features = config.get('categorical_features', [])
        self.target_col = 'target'
        self.label_encoders = {}
        self.best_threshold = 0.5
        
        # Create results directory
        self.results_dir = Path(__file__).parent.parent / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self, X_train, X_test, y_train, y_test):
        """Prepare data for TabTransformer (encode categoricals, normalize numericals)"""
        print("\nPreparing data for TabTransformer...")
        
        # Remove non-feature columns
        cols_to_drop = ['issue_d', 'issue_date', 'issue_year']
        X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns], errors='ignore')
        X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], errors='ignore')
        
        # Identify numerical and categorical columns
        if not self.numerical_features:
            self.numerical_features = X_train.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8']).columns.tolist()
        
        if not self.categorical_features:
            self.categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"  Numerical features: {len(self.numerical_features)}")
        print(f"  Categorical features: {len(self.categorical_features)}")
        
        # Handle NaN in numerical features
        X_train_num = X_train[self.numerical_features].copy()
        X_test_num = X_test[self.numerical_features].copy()
        
        for col in self.numerical_features:
            median_val = X_train_num[col].median()
            X_train_num[col] = X_train_num[col].fillna(median_val)
            X_test_num[col] = X_test_num[col].fillna(median_val)
        
        # Encode categorical features
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_test_cat = pd.DataFrame(index=X_test.index)
        self.categorical_dims = []
        
        for col in self.categorical_features:
            le = LabelEncoder()
            X_train_cat[col] = le.fit_transform(X_train[col].fillna('NA').astype(str))
            X_test_col = X_test[col].fillna('NA').astype(str)
            X_test_cat[col] = X_test_col.apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
            self.label_encoders[col] = le
            self.categorical_dims.append(len(le.classes_))
            print(f"    {col}: {len(le.classes_)} categories")
        
        # Normalize numerical features
        scaler = StandardScaler()
        X_train_num_scaled = scaler.fit_transform(X_train_num)
        X_test_num_scaled = scaler.transform(X_test_num)
        self.scaler = scaler
        
        # Calculate class weights for imbalanced data
        n_samples = len(y_train)
        n_pos = y_train.sum()
        n_neg = n_samples - n_pos
        
        # Weight for positive class (defaults) - used in BCELoss
        self.pos_weight = torch.FloatTensor([n_neg / n_pos]).to(self.device) if n_pos > 0 else torch.FloatTensor([1.0]).to(self.device)
        print(f"\n  Class imbalance ratio: {n_neg/n_pos:.2f}:1 (negative:positive)")
        print(f"  Positive class weight: {self.pos_weight.item():.2f}")
        
        # Convert to tensors
        X_train_num_tensor = torch.FloatTensor(X_train_num_scaled).to(self.device)
        X_test_num_tensor = torch.FloatTensor(X_test_num_scaled).to(self.device)
        X_train_cat_tensor = torch.LongTensor(X_train_cat.values).to(self.device)
        X_test_cat_tensor = torch.LongTensor(X_test_cat.values).to(self.device)
        
        # Changed: use FloatTensor for BCE loss
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_num_tensor, X_train_cat_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_num_tensor, X_test_cat_tensor, y_test_tensor)
        
        batch_size = int(self.config.get('batch_size', 256))
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"  Train batches: {len(self.train_loader)}, Test batches: {len(self.test_loader)}")
        return self.train_loader, self.test_loader
    
    def build_model(self):
        """Build TabTransformer model"""
        model_config = self.config.get('model', {})
        
        if len(self.categorical_features) == 0:
            raise ValueError("TabTransformer requires at least 1 categorical feature")
        
        model = TabTransformer(
            num_numerical_features=len(self.numerical_features),
            num_categorical_features=len(self.categorical_features),
            categorical_dims=self.categorical_dims,
            embedding_dim=int(model_config.get('embedding_dim', 32)),
            depth=int(model_config.get('depth', 4)),
            heads=min(int(model_config.get('heads', 4)), len(self.categorical_features)),
            dim_head=int(model_config.get('dim_head', 32)),
            mlp_dim=int(model_config.get('mlp_dim', 256)),
            num_classes=1,  # Single probability output
            dropout=float(model_config.get('dropout', 0.2))
        ).to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model created with {num_params:,} parameters")
        
        return model
    
    def train_epoch(self, model, optimizer, criterion, epoch):
        """Train one epoch"""
        model.train()
        total_loss = 0
        
        for batch_idx, (X_num, X_cat, y) in enumerate(self.train_loader):
            optimizer.zero_grad()
            probs = model(X_num, X_cat)  # Now outputs probability directly
            loss = criterion(probs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def find_optimal_threshold(self, labels, probs):
        """Find optimal threshold using precision-recall curve"""
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        
        # Find threshold that maximizes F1
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            return thresholds[best_idx]
        return 0.5
    
    def evaluate(self, model, find_threshold=False):
        """Evaluate model on test set"""
        model.eval()
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for X_num, X_cat, y in self.test_loader:
                probs = model(X_num, X_cat)  # Direct probability output (0-1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold if requested
        if find_threshold:
            self.best_threshold = self.find_optimal_threshold(all_labels, all_probs)
        
        # Apply threshold
        all_preds = (all_probs >= self.best_threshold).astype(int)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        mae = mean_absolute_error(all_labels, all_probs)
        
        # Handle edge cases in confusion matrix
        cm = confusion_matrix(all_labels.astype(int), all_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'auc': auc, 
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'mae': mae,
            'threshold': self.best_threshold,
            'true_positives': int(tp), 
            'true_negatives': int(tn),
            'false_positives': int(fp), 
            'false_negatives': int(fn)
        }, all_probs
    
    def train(self, X_train, X_test, y_train, y_test):
        """Full training pipeline with class weighting and threshold tuning"""
        self.prepare_data(X_train, X_test, y_train, y_test)
        model = self.build_model()
        
        # Convert config values
        learning_rate = float(self.config.get('learning_rate', 0.0005))
        weight_decay = float(self.config.get('weight_decay', 0.01))
        epochs = int(self.config.get('epochs', 100))
        patience = int(self.config.get('early_stopping_patience', 15))
        
        # Use BCELoss with pos_weight for class imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        # Note: Since model outputs sigmoid already, we use BCELoss instead
        criterion = nn.BCELoss(reduction='none')
        
        # Custom weighted BCE loss function
        def weighted_bce_loss(probs, targets):
            bce = criterion(probs, targets)
            weights = torch.where(targets == 1, self.pos_weight, torch.ones_like(targets))
            return (bce * weights).mean()
        
        # Use AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        best_auc = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_auc': [], 'val_f1': [], 'val_recall': []}
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        print(f"  Using weighted BCE loss for probability output (0-1)")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Train epoch with weighted BCE
            model.train()
            total_loss = 0
            for X_num, X_cat, y in self.train_loader:
                optimizer.zero_grad()
                probs = model(X_num, X_cat)
                loss = weighted_bce_loss(probs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(self.train_loader)
            
            scheduler.step()
            
            # Find optimal threshold every 5 epochs
            find_thresh = (epoch % 5 == 0)
            metrics, _ = self.evaluate(model, find_threshold=find_thresh)
            
            history['train_loss'].append(train_loss)
            history['val_auc'].append(metrics['auc'])
            history['val_f1'].append(metrics['f1'])
            history['val_recall'].append(metrics['recall'])
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} (LR: {current_lr:.6f})")
            print(f"  AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
            print(f"  Threshold: {metrics['threshold']:.3f} | TP: {metrics['true_positives']} | FP: {metrics['false_positives']}")
            
            # Use F1 score for early stopping (better for imbalanced data)
            score = metrics['auc'] + metrics['f1']  # Combined metric
            if score > best_auc:
                best_auc = score
                patience_counter = 0
                torch.save(model.state_dict(), self.results_dir / 'best_tab_transformer.pth')
                print(f"  ✓ Best model saved (AUC+F1: {score:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping after {epoch+1} epochs")
                break
        
        # Load best model and find final optimal threshold
        model.load_state_dict(torch.load(self.results_dir / 'best_tab_transformer.pth', weights_only=True))
        final_metrics, probs = self.evaluate(model, find_threshold=True)
        
        print("\n" + "=" * 60)
        print("FINAL TEST SET RESULTS (Probability Output: 0-1)")
        print("=" * 60)
        print(f"  Optimal Threshold: {final_metrics['threshold']:.3f}")
        print(f"  AUC-ROC:     {final_metrics['auc']:.4f}")
        print(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
        print(f"  Precision:   {final_metrics['precision']:.4f}")
        print(f"  Recall:      {final_metrics['recall']:.4f}")
        print(f"  F1-Score:    {final_metrics['f1']:.4f}")
        print(f"  MAE:         {final_metrics['mae']:.4f}")
        print("-" * 60)
        print("Confusion Matrix:")
        print(f"  TP: {final_metrics['true_positives']:,} | FP: {final_metrics['false_positives']:,}")
        print(f"  FN: {final_metrics['false_negatives']:,} | TN: {final_metrics['true_negatives']:,}")
        print("=" * 60)
        
        # Save history
        with open(self.results_dir / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        return model, final_metrics
