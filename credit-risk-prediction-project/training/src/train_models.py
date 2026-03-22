"""
train_models.py - Model training with XGBoost, LR, RF, Gaussian Naive Bayes
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-optimize, but make it optional
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization will be skipped.")
    print("Install with: pip install scikit-optimize")
    # Define dummy classes to avoid errors
    class BayesSearchCV:
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-optimize not installed")
    class Real:
        pass
    class Integer:
        pass
    class Categorical:
        pass

class ModelTrainer:
    """Model training with paper's models + improvements"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def train_comparison_models(self, X_train, X_val, y_train, y_val, feature_set_name):
        """
        Train and compare multiple models:
        - XGBoost (paper's feature selection method)
        - Logistic Regression (with LASSO as suggested in future work)
        - Random Forest
        - Gaussian Naive Bayes (simple baseline)
        """
        
        print(f"\n{'='*60}")
        print(f"Training models with {feature_set_name} features")
        print(f"{'='*60}")
        
        # Calculate scale_pos_weight for XGBoost (must be float, not string)
        n_negative = len(y_train[y_train == 0])
        n_positive = len(y_train[y_train == 1])
        scale_pos_weight = n_negative / max(n_positive, 1)  # Avoid division by zero
        print(f"  Class imbalance: {n_negative}:{n_positive} (ratio: {scale_pos_weight:.2f})")
        
        # Get model configs
        lr_config = self.config['models']['logistic_regression'].copy()
        rf_config = self.config['models']['random_forest'].copy()
        xgb_config = self.config['models']['xgboost'].copy()
        nb_config = self.config['models']['gaussian_nb'].copy()
        
        # Update XGBoost config with calculated weight
        xgb_config.update({
            'scale_pos_weight': scale_pos_weight,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        
        # Update LR config for LASSO (paper's suggestion)
        lr_config.update({
            'penalty': 'l1',  # LASSO for feature selection
            'solver': 'saga'  # Supports L1 regularization
        })
        
        # Define models
        models_config = {
            'XGBoost': XGBClassifier(**xgb_config),
            'Logistic Regression (LASSO)': LogisticRegression(**lr_config),
            'Random Forest': RandomForestClassifier(**rf_config),
            'Gaussian Naive Bayes': GaussianNB(**nb_config)
        }
        
        results = []
        
        for name, model in models_config.items():
            print(f"\n{name}:")
            start_time = time.time()
            
            try:
                # Train model
                if name == 'XGBoost':
                    # Try with early stopping, fallback if not supported
                    try:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False,
                            early_stopping_rounds=50
                        )
                    except TypeError as e:
                        # If early_stopping_rounds not supported, train normally
                        print(f"  Note: Using standard fit (early stopping not available)")
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                train_time = time.time() - start_time
                
                # Predict
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics['train_time'] = train_time
                metrics['model_name'] = name
                metrics['feature_set'] = feature_set_name
                
                results.append(metrics)
                
                # Store model
                self.models[f"{name}_{feature_set_name}"] = model
                
                print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Time:    {train_time:.2f}s")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)[:100]}...")
                print(f"  Trying with simplified parameters...")
                
                # Fallback to simpler model
                try:
                    if name == 'Logistic Regression (LASSO)':
                        model = LogisticRegression(
                            max_iter=1000,
                            class_weight='balanced',
                            random_state=42
                        )
                    elif name == 'Random Forest':
                        model = RandomForestClassifier(
                            n_estimators=100,
                            class_weight='balanced',
                            random_state=42,
                            n_jobs=-1
                        )
                    elif name == 'XGBoost':
                        model = XGBClassifier(
                            n_estimators=100,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='logloss',
                            scale_pos_weight=scale_pos_weight
                        )
                    elif name == 'Gaussian Naive Bayes':
                        model = GaussianNB()
                    
                    model.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                    metrics['train_time'] = train_time
                    metrics['model_name'] = name
                    metrics['feature_set'] = feature_set_name
                    
                    results.append(metrics)
                    self.models[f"{name}_{feature_set_name}"] = model
                    
                    print(f"  ✓ Trained with defaults")
                    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                    print(f"  Time:    {train_time:.2f}s")
                    
                except Exception as e2:
                    print(f"  ✗ Could not train {name}: {str(e2)[:100]}...")
                    continue
        
        return pd.DataFrame(results)
    
    def optimize_xgboost(self, X_train, X_val, y_train, y_val):
        """
        Bayesian optimization for XGBoost (improvement over grid search)
        Falls back to random search if scikit-optimize not available
        """
        optimization_method = self.config['optimization']['method']
        
        if optimization_method == 'bayesian' and not SKOPT_AVAILABLE:
            print("\nWarning: scikit-optimize not installed for Bayesian optimization.")
            print("Falling back to random search...")
            optimization_method = 'random'
        
        # Calculate scale_pos_weight
        scale_pos_weight = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
        
        if optimization_method == 'bayesian':
            print("\nPerforming Bayesian optimization for XGBoost...")
            
            # Define search space
            search_spaces = {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0, 5),
                'reg_alpha': Real(0, 10),
                'reg_lambda': Real(0, 10)
            }
            
            # Base model
            base_model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False,
                tree_method='hist'
            )
            
            # Bayesian optimization
            opt = BayesSearchCV(
                base_model,
                search_spaces,
                n_iter=self.config['optimization']['n_iter'],
                cv=self.config['optimization']['cv'],
                scoring=self.config['optimization']['scoring'],
                random_state=42,
                verbose=0,
                n_jobs=-1
            )
            
            opt.fit(X_train, y_train)
            
            print(f"✓ Best parameters found:")
            for param, value in opt.best_params_.items():
                print(f"  {param}: {value}")
            print(f"  Best score: {opt.best_score_:.4f}")
            
            return opt.best_estimator_
        
        elif optimization_method == 'random':
            print("\nPerforming random search optimization for XGBoost...")
            from sklearn.model_selection import RandomizedSearchCV
            
            # Parameter grid for random search
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 5, 7, 9, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 1, 3, 5],
                'reg_alpha': [0, 1, 5, 10],
                'reg_lambda': [0, 1, 5, 10]
            }
            
            # Base model
            base_model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False,
                tree_method='hist'
            )
            
            # Random search
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=param_dist,
                n_iter=self.config['optimization']['n_iter'],
                cv=self.config['optimization']['cv'],
                scoring=self.config['optimization']['scoring'],
                random_state=42,
                verbose=0,
                n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            
            print(f"✓ Best parameters found:")
            for param, value in random_search.best_params_.items():
                print(f"  {param}: {value}")
            print(f"  Best score: {random_search.best_score_:.4f}")
            
            return random_search.best_estimator_
        
        else:
            print("\nUsing default XGBoost parameters...")
            # Use default model with basic settings
            model = XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False,
                tree_method='hist'
            )
            model.fit(X_train, y_train)
            return model
    
    def feature_selection_xgboost(self, X, y, feature_names):
        """
        Feature selection using XGBoost importance (paper's method)
        Returns top k features based on importance
        """
        print("\nPerforming feature selection with XGBoost (paper's method)...")
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False
        )
        
        model.fit(X, y)
        
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Display top features
        print("Top 20 features by importance:")
        for i in range(min(20, len(feature_names))):
            print(f"{i+1:2d}. {feature_names[indices[i]]:30s} {importances[indices[i]]:.6f}")
        
        # Return indices of top k features
        return indices
    
    def business_profit_analysis(self, model, X_test, y_test, loan_data=None, thresholds=None):
        """
        Analyze business profit for different thresholds using per-loan calculations.

        Profit logic:
          TP (correctly rejected bad loan)  = + loan_amnt * loss_given_default  (principal saved)
          FN (missed bad loan, approved it) = - loan_amnt * loss_given_default  (principal lost)
          FP (wrongly rejected good loan)   = - loan_amnt * int_rate * term     (interest lost)
          TN (correctly approved good loan) = + loan_amnt * int_rate * term     (interest earned)
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test = np.array(y_test)

        business_config = self.config['business']
        lgd = business_config['loss_given_default']
        default_term = business_config['default_term_years']

        # Build per-loan value arrays
        if loan_data is not None and 'loan_amnt' in loan_data.columns and 'int_rate' in loan_data.columns:
            loan_amnt = np.array(loan_data['loan_amnt'].fillna(loan_data['loan_amnt'].median()))
            int_rate = np.array(loan_data['int_rate'].fillna(loan_data['int_rate'].median()))
            # int_rate is stored as percentage (e.g. 13.5), convert to decimal
            int_rate = int_rate / 100.0
            term_years = np.full(len(loan_amnt), default_term)
            if 'term' in loan_data.columns:
                term_years = np.array(loan_data['term'].fillna(36)) / 12.0
        else:
            # Fallback to dataset averages if loan data not provided
            loan_amnt = np.full(len(y_test), 14000.0)
            int_rate  = np.full(len(y_test), 0.135)
            term_years = np.full(len(y_test), default_term)

        # Per-loan profit values
        principal_value  = loan_amnt * lgd                   # what we save/lose on a default
        interest_value   = loan_amnt * int_rate * term_years # what we earn/lose on a good loan

        profits = []
        metrics_list = []

        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)

            tp_mask = (y_pred == 1) & (y_test == 1)  # correctly rejected bad loan
            fn_mask = (y_pred == 0) & (y_test == 1)  # missed bad loan (approved it)
            fp_mask = (y_pred == 1) & (y_test == 0)  # wrongly rejected good loan
            tn_mask = (y_pred == 0) & (y_test == 0)  # correctly approved good loan

            profit = (
                  principal_value[tp_mask].sum()   # saved principal on true bad loans
                - principal_value[fn_mask].sum()   # lost principal on missed bad loans
                - interest_value[fp_mask].sum()    # lost interest on wrongly rejected good loans
                + interest_value[tn_mask].sum()    # earned interest on approved good loans
            )

            tp = tp_mask.sum(); tn = tn_mask.sum()
            fp = fp_mask.sum(); fn = fn_mask.sum()

            metrics = {
                'threshold': thresh,
                'profit': profit,
                'accuracy': (tp + tn) / len(y_test),
                'precision': tp / max((tp + fp), 1),
                'recall': tp / max((tp + fn), 1),
                'f1_score': 2 * tp / max(2 * tp + fp + fn, 1),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }

            profits.append(profit)
            metrics_list.append(metrics)

        # Find optimal threshold
        best_idx = np.argmax(profits)
        best_threshold = thresholds[best_idx]
        best_metrics = metrics_list[best_idx]

        return best_threshold, best_metrics, pd.DataFrame(metrics_list)
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        return {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': self._calculate_pr_auc(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': self._calculate_precision(y_true, y_pred),
            'recall': self._calculate_recall(y_true, y_pred)
        }
    
    def _calculate_pr_auc(self, y_true, y_pred_proba):
        """Calculate precision-recall AUC"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall, precision)
    
    def _calculate_precision(self, y_true, y_pred):
        """Calculate precision"""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        return tp / max(tp + fp, 1)
    
    def _calculate_recall(self, y_true, y_pred):
        """Calculate recall"""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return tp / max(tp + fn, 1)