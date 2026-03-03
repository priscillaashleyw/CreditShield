"""
config.py - Configuration for the credit risk prediction project
Paper: "Credit scoring for peer-to-peer lending using machine learning techniques"
       Quantitative Finance and Economics, Volume 6, Issue 2, 303-325 (2022)
"""

CONFIG = {
    # Paths and directories
    'paths': {
        'raw_data': 'kaggle://wordsforthewise/lending-club/accepted_2007_to_2018Q4.csv',
        'models_dir': 'models',
        'results_dir': 'results',
        'logs_dir': 'logs',
        'visualizations_dir': 'results'
    },
    
    # Data settings (following paper's methodology)
    'data_settings': {
        # Paper uses data from 2013-2014 (3-year data for 2-year period)
        'years': [2013, 2014],
        
        # Use all data (None) or sample for testing (e.g., 100000)
        'sample_size': None,
        
        # Essential columns to load (based on paper's feature selection; removed leakage columns)
        'essential_columns': [
            # Loan information
            'loan_amnt', 'int_rate', 'term', 'emp_title', 'emp_length', 'home_ownership', 
            'annual_inc', 'verification_status', 'issue_d', 'loan_status', 'purpose', 
            'title', 'zip_code', 'addr_state',
            
            # Financial ratios
            'dti', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths',
            'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 
            'initial_list_status',
            
            # FICO scores (paper's most important features)
            'last_fico_range_high', 'last_fico_range_low',
            
            # Credit history features (from paper's Table 3)
            'collections_12_mths_ex_med', 'mths_since_last_major_derog',
            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal',
            'total_rev_hi_lim', 'avg_cur_bal', 'bc_open_to_buy',
            'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt',
            'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
            'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc',
            'mths_since_recent_bc', 'mths_since_recent_inq',
            'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
            'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
            'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
            'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
            'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
            'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
            'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit'
        ],

        # Leaked columns to remove
        'leaked_columns': [
            'total_pymnt', 'total_pymnt_inv', 'recoveries', 'last_pymnt_amnt', 
            'last_pymnt_d', 'collection_recovery_fee', 'out_prncp', 'out_prncp_inv',
            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'grade', 'sub_grade'
        ],

        
        # Paper filters out incomplete loan statuses
        'incomplete_statuses': [
            'Current', 'In Grace Period', 'Late (16-30 days)',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        ]
    },
    
    # Target definition strategies
    'target_settings': {
        'default_strategy': 'business',  # 'paper', 'business', 'conservative'
        
        # Paper's definition (includes late payments)
        'paper': {
            'default_statuses': ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)'],
            'good_statuses': ['Fully Paid']
        },
        
        # Business-oriented definition (excludes minor lates)
        'business': {
            'default_statuses': ['Charged Off', 'Default', 'Late (31-120 days)'],
            'good_statuses': ['Fully Paid']
        },
        
        # Conservative definition (only charge-offs and defaults)
        'conservative': {
            'default_statuses': ['Charged Off', 'Default'],
            'good_statuses': ['Fully Paid']
        }
    },
    
    # Feature sets for comparison
    'feature_sets': {
        # Paper's 16 features from Table 5
        'paper_16': [
            'dti', 'mo_sin_old_il_acct', 'annual_inc', 'revol_util',
            'avg_cur_bal', 'total_bc_limit', 'mo_sin_old_rev_tl_op',
            'revol_bal', 'total_bal_ex_mort', 'mths_since_recent_bc',
            'total_acc', 'fico_range_low', 'last_fico_range_high',
            'mths_since_recent_inq', 'mo_sin_rcnt_rev_tl_op', 'pct_tl_nvr_dlq'
        ],
        
        # Paper's basic features from methodology
        'paper_basic': [
            'loan_amnt', 'int_rate', 'grade_numeric', 'emp_length_numeric',
            'annual_inc', 'dti', 'revol_util_decimal', 'delinq_2yrs',
            'inq_last_6mths', 'open_acc', 'total_acc'
        ],
        
        # Reference date for credit history (September 2018 as per paper)
        'reference_date': '2018-09-01'
    },
    
    # Model configurations - UPDATED: XGBoost, LR, RF, Gaussian Naive Bayes
    'models': {
        # XGBoost (paper's feature selection method + our improvements)
        'xgboost': {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 50
            # scale_pos_weight will be calculated dynamically
        },
        
        # Random Forest
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced_subsample',  # Cost-sensitive learning
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        },
        
        # Logistic Regression with LASSO (paper's suggestion for future work)
        'logistic_regression': {
            'C': 0.1,
            'max_iter': 1000,
            'class_weight': 'balanced',  # Cost-sensitive learning
            'random_state': 42,
            'solver': 'saga',  # Supports L1 regularization
            'penalty': 'l1',  # LASSO for feature selection
            'tol': 1e-4,
            'verbose': 0
        },
        
        # Gaussian Naive Bayes (simple baseline model)
        'gaussian_nb': {
            'var_smoothing': 1e-9  # Small value to avoid zero variance
        }
    },
    
    # Training settings
    'training': {
        # Paper uses random split, we use time-based as improvement
        'test_size': 0.2,
        'val_size': 0.15,  # For time-based split
        'time_split': True,  # Use time-based split (True) or random split (False)
        'random_state': 42,
        'stratify': True,  # Maintain class distribution in splits
        
        # Cross-validation settings
        'cv_folds': 5,
        'cv_shuffle': True,
        
        # Time-series cross-validation for time-based split
        'ts_cv_splits': 3,
        'ts_cv_test_size': 0.2
    },
    
    # Optimization settings
    'optimization': {
        'method': 'bayesian',  # 'grid', 'random', 'bayesian' (our improvement)
        'n_iter': 30,  # Reduced from 50 for faster training
        'cv': 3,  # Cross-validation folds during optimization
        'scoring': 'roc_auc',  # Primary metric for optimization
        
        # XGBoost search space for Bayesian optimization
        'xgboost_search_space': {
            'n_estimators': (100, 500),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'gamma': (0, 5),
            'reg_alpha': (0, 10),
            'reg_lambda': (0, 10),
            'min_child_weight': (1, 10)
        }
    },
    
    # Business settings for profit analysis (improvement over accuracy-only)
    'business': {
        # Profit/loss values (example - should be calibrated to your business)
        'profit_tp': 3000,    # Profit from true positive (correctly approved good loan)
        'profit_tn': 1000,    # Profit from true negative (avoided bad loan)
        'loss_fp': -15000,    # Loss from false positive (approved bad loan)
        'loss_fn': -5000,     # Loss from false negative (rejected good loan)
        
        # Threshold search range
        'threshold_min': 0.1,
        'threshold_max': 0.9,
        'threshold_step': 0.05,
        
        # Risk categories for interpretation
        'risk_categories': {
            'low_risk': (0.0, 0.2),
            'medium_risk': (0.2, 0.4),
            'high_risk': (0.4, 0.6),
            'very_high_risk': (0.6, 1.0)
        }
    },
    
    # Feature engineering settings
    'feature_engineering': {
        # Text feature extraction from loan titles (paper's future work suggestion)
        'text_features': {
            'extract': True,
            'keywords': ['debt', 'consolidation', 'credit', 'card', 'medical', 
                        'home', 'car', 'business', 'vacation', 'wedding'],
            'min_title_length': 3,
            'max_title_length': 100
        },
        
        # Interaction terms (our improvement)
        'interaction_terms': [
            ('grade_numeric', 'dti'),
            ('loan_amnt', 'annual_inc'),
            ('int_rate', 'loan_amnt'),
            ('emp_length_numeric', 'annual_inc')
        ],
        
        # Financial ratios (our improvement)
        'financial_ratios': {
            'loan_to_income': True,
            'payment_to_income': True,
            'debt_to_income_enhanced': True,
            'credit_utilization_ratio': True
        },
        
        # Missing value handling (following paper's methodology)
        'missing_values': {
            # For features where NA means "no occurrence"
            'na_means_none': ['mths_since_last_delinq', 'mths_since_last_record',
                             'mths_since_last_major_derog', 'mths_since_recent_bc'],
            'fill_value': -1,  # As per paper's methodology
            
            # For other features, use median imputation
            'default_strategy': 'median',
            
            # Create missing indicators as per paper
            'create_indicators': True
        }
    },
    
    # Evaluation metrics
    'evaluation': {
        'primary_metric': 'roc_auc',
        'secondary_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc_pr'],
        
        # Business metrics (our improvement)
        'business_metrics': ['profit', 'expected_value', 'return_on_investment'],
        
        # Statistical significance testing
        'statistical_tests': {
            'perform': True,
            'confidence_level': 0.95,
            'n_resamples': 1000
        }
    },
    
    # Visualization settings
    'visualization': {
        'save_plots': True,
        'dpi': 150,
        'format': 'png',
        'style': 'seaborn-v0_8-darkgrid',
        
        # Color scheme
        'colors': {
            'paper': '#1f77b4',      # Blue
            'our': '#ff7f0e',        # Orange
            'baseline': '#2ca02c',   # Green
            'optimal': '#d62728',    # Red
            'train': '#9467bd',      # Purple
            'val': '#8c564b',        # Brown
            'test': '#e377c2'        # Pink
        },
        
        # Plot sizes
        'figure_sizes': {
            'small': (8, 6),
            'medium': (12, 8),
            'large': (16, 12),
            'comparison': (15, 10)
        },
        
        # Plots to generate
        'plots': {
            'model_comparison': True,
            'feature_importance': True,
            'profit_analysis': True,
            'roc_curves': True,
            'precision_recall_curves': True,
            'threshold_analysis': True,
            'learning_curves': True,
            'calibration_curves': True
        }
    }
}