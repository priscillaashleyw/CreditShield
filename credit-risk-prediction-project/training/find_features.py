# training/find_features.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_training_features():
    """Analyze what features were actually used in training"""
    
    print("🔍 Analyzing training features...")
    print("=" * 60)
    
    # Check your latest results file
    results_dir = Path("results")
    csv_files = list(results_dir.glob("model_comparison_*.csv"))
    
    if csv_files:
        latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
        print(f"Latest results file: {latest_csv.name}")
        
        df_results = pd.read_csv(latest_csv)
        print(f"\nResults columns: {list(df_results.columns)}")
        
        # Check feature sets used
        if 'feature_set' in df_results.columns:
            print(f"\nFeature sets used in training:")
            for feature_set in df_results['feature_set'].unique():
                subset = df_results[df_results['feature_set'] == feature_set]
                n_features = subset['n_features'].iloc[0] if 'n_features' in subset.columns else 'Unknown'
                print(f"  - {feature_set}: {n_features} features")
    
    print("\n" + "=" * 60)
    
    # Based on your training output, you had:
    # - paper_16: 15 features
    # - our_enhanced: 40 features
    
    print("\n📋 Based on your training output:")
    print("paper_16 features (15 features):")
    paper_16 = [
        'dti', 'mo_sin_old_il_acct', 'annual_inc', 'revol_util',
        'avg_cur_bal', 'total_bc_limit', 'mo_sin_old_rev_tl_op',
        'revol_bal', 'total_bal_ex_mort', 'mths_since_recent_bc',
        'total_acc', 'fico_range_low', 'last_fico_range_high',
        'mths_since_recent_inq', 'mo_sin_rcnt_rev_tl_op', 'pct_tl_nvr_dlq'
    ]
    print(f"  {len(paper_16)} features from paper")
    
    print("\nour_enhanced features (40 features likely include):")
    enhanced = paper_16 + [
        'loan_amnt', 'int_rate', 'grade_numeric', 'emp_length_numeric',
        'revol_util_decimal', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
        'loan_to_income', 'int_rate_times_loan', 'subprime_high_dti',
        'has_delinquencies', 'title_length', 'title_word_count'
    ]
    print(f"  Paper's 16 + ~24 engineered features = ~40 total")
    
    print("\n" + "=" * 60)
    
    # Let's check what's in your config
    try:
        from config import CONFIG
        print(f"\nConfig feature sets:")
        for key, features in CONFIG['feature_sets'].items():
            if isinstance(features, list):
                print(f"  {key}: {len(features)} features")
            else:
                print(f"  {key}: {features}")
    except:
        print("\nCould not load config")
    
    return enhanced

if __name__ == "__main__":
    features = analyze_training_features()
    
    # Save to file
    output_path = Path("../deployment/model_artifacts/training_features.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({'enhanced_features': features}, f, indent=2)
    
    print(f"\n✅ Saved feature list to: {output_path}")