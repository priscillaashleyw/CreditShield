"""
train.py - Complete training pipeline with paper methodology + improvements
Run with: python train.py
For quick test: python train.py --quick
For full training: python train.py --full
For custom sample size: python train.py --sample 100000
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Parse command line arguments FIRST
parser = argparse.ArgumentParser(description='Credit Risk Prediction Training')
parser.add_argument('--quick', action='store_true', 
                   help='Quick test with 50,000 samples (default)')
parser.add_argument('--full', action='store_true', 
                   help='Full training with all data (no sampling)')
parser.add_argument('--sample', type=int, 
                   help='Custom sample size (e.g., --sample 100000)')
parser.add_argument('--no-optimize', action='store_true',
                   help='Skip Bayesian optimization to save time')
parser.add_argument('--no-viz', action='store_true',
                   help='Skip visualization generation')
args = parser.parse_args()

# Import modules AFTER parsing arguments
from config import CONFIG
from src.load_data import DataLoader
from src.build_features import FeatureEngineer
from src.train_models import ModelTrainer

print("=" * 80)
print("CREDIT RISK PREDICTION - PAPER REPLICATION + IMPROVEMENTS")
print("=" * 80)
print(f"Paper: Quantitative Finance and Economics, Volume 6, Issue 2")
print(f"Improvements: Cost-sensitive learning, Time-based split, Business profit analysis")
print("=" * 80)

# Apply command-line arguments to config
if args.full:
    CONFIG['data_settings']['sample_size'] = None
    CONFIG['optimization']['n_iter'] = 30
    CONFIG['training']['time_split'] = True
    print("🚀 MODE: FULL TRAINING (all data, time-based split)")
elif args.sample:
    CONFIG['data_settings']['sample_size'] = args.sample
    CONFIG['optimization']['n_iter'] = 10
    CONFIG['training']['time_split'] = False
    print(f"📊 MODE: CUSTOM SAMPLING ({args.sample:,} samples, random split)")
elif args.quick:
    CONFIG['data_settings']['sample_size'] = 50000
    CONFIG['optimization']['n_iter'] = 5
    CONFIG['training']['time_split'] = False
    print("⚡ MODE: QUICK TEST (50,000 samples, random split)")
else:
    # DEFAULT: Quick mode (to avoid hanging)
    CONFIG['data_settings']['sample_size'] = 50000
    CONFIG['optimization']['n_iter'] = 10
    CONFIG['training']['time_split'] = False
    print("🔧 MODE: DEFAULT (50,000 samples, random split)")
    print("   Use --full for complete training or --sample N for custom size")

if args.no_optimize:
    CONFIG['optimization']['method'] = 'random'
    print("⚠️  Skipping Bayesian optimization (using random search)")

if args.no_viz:
    print("⚠️  Skipping visualization generation")

print(f"Sample size: {CONFIG['data_settings']['sample_size'] or 'ALL DATA'}")
print(f"Optimization iterations: {CONFIG['optimization']['n_iter']}")
print(f"Time-based split: {CONFIG['training']['time_split']}")
print("=" * 80)

def main():
    """Main training pipeline"""
    
    # Create directories
    for dir_path in ['models', 'results', 'logs']:
        Path(dir_path).mkdir(exist_ok=True)
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    feature_engineer = FeatureEngineer(CONFIG)
    model_trainer = ModelTrainer(CONFIG)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. LOAD AND PREPARE DATA
    print("\n1. LOADING DATA")
    print("-" * 40)
    
    try:
        df = data_loader.load_and_filter_data()
        print(f"✓ Data loaded: {len(df):,} rows")
        
        df = data_loader.define_target(df, strategy='business')
        print(f"✓ Target defined: {df['target'].mean():.2%} default rate")
        
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return None, None
    
    # Save data info
    try:
        data_info = {
            'total_rows': int(len(df)),
            'default_rate': float(df['target'].mean()),
            'n_defaults': int(df['target'].sum()),
            'n_good': int(len(df) - df['target'].sum()),
            'date_range': {
                'min': df['issue_date'].min().strftime('%Y-%m-%d') if not pd.isna(df['issue_date'].min()) else None,
                'max': df['issue_date'].max().strftime('%Y-%m-%d') if not pd.isna(df['issue_date'].max()) else None
            },
            'sample_size_used': CONFIG['data_settings']['sample_size'],
            'training_mode': 'full' if CONFIG['data_settings']['sample_size'] is None else 'sampled'
        }
        
        with open(f'results/data_info_{timestamp}.json', 'w') as f:
            json.dump(data_info, f, indent=2)
        
        print(f"\n✓ Data info saved: results/data_info_{timestamp}.json")
        
    except Exception as e:
        print(f"⚠️  Could not save data info: {e}")
    
    # 2. CREATE FEATURES
    print("\n\n2. FEATURE ENGINEERING")
    print("-" * 40)
    
    try:
        df = feature_engineer.create_paper_features(df)
        feature_sets = feature_engineer.create_feature_sets(df)
        print(f"✓ Created features, total columns: {df.shape[1]}")
        
        # Remove 'all' features if present (too many)
        if 'all' in feature_sets and len(feature_sets['all']) > 100:
            print(f"⚠️  Skipping 'all' features ({len(feature_sets['all'])} features - too many)")
            del feature_sets['all']
            
    except Exception as e:
        print(f"❌ ERROR in feature engineering: {e}")
        return None, None
    
    # 3. SPLIT DATA
    print("\n\n3. DATA SPLITTING")
    print("-" * 40)
    
    try:
        if CONFIG['training']['time_split']:
            print("Using time-based split...")
            train_df, val_df, test_df = data_loader.time_based_split(
                df,
                train_ratio=1 - CONFIG['training']['test_size'] - CONFIG['training']['val_size'],
                val_ratio=CONFIG['training']['val_size']
            )
        else:
            print("Using random split...")
            X_train, X_test, y_train, y_test = data_loader.random_split(
                df,
                test_size=CONFIG['training']['test_size'],
                random_state=CONFIG['training']['random_state']
            )
            
            # Convert to DataFrames
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train
            test_df = pd.DataFrame(X_test)
            test_df['target'] = y_test
            val_df = test_df.copy()  # Use test as validation for random split
        
        print(f"✓ Data split:")
        print(f"  Training: {len(train_df):,} samples")
        print(f"  Validation: {len(val_df):,} samples")
        print(f"  Testing: {len(test_df):,} samples")
        
    except Exception as e:
        print(f"❌ ERROR in data splitting: {e}")
        return None, None
    
    # 4. TRAIN MODELS
    print("\n\n4. TRAINING MODELS")
    print("-" * 40)
    
    all_results = []
    trained_feature_sets = []
    
    # Train with paper_16 and our_enhanced features only
    for set_name in ['paper_16', 'our_enhanced']:
        if set_name not in feature_sets:
            continue
            
        features = feature_sets[set_name]
        if len(features) < 5:
            print(f"\nSkipping {set_name} (only {len(features)} features)")
            continue
            
        print(f"\n🔧 Training with {set_name} features ({len(features)} features)")
        
        try:
            # Prepare features
            X_train_scaled, imputer, scaler = feature_engineer.prepare_features(train_df, features)
            X_val_scaled = scaler.transform(imputer.transform(val_df[features]))
            
            y_train = train_df['target'].values
            y_val = val_df['target'].values
            
            # Train models
            results_df = model_trainer.train_comparison_models(
                X_train_scaled, X_val_scaled, y_train, y_val, set_name
            )
            
            results_df['n_features'] = len(features)
            all_results.append(results_df)
            trained_feature_sets.append(set_name)
            
            # Save imputer and scaler
            joblib.dump(imputer, f'models/imputer_{set_name}_{timestamp}.pkl')
            joblib.dump(scaler, f'models/scaler_{set_name}_{timestamp}.pkl')
            
            print(f"✓ {set_name} features trained successfully")
            
        except Exception as e:
            print(f"✗ Error with {set_name}: {str(e)[:80]}...")
            continue
    
    # Check if we have results
    if not all_results:
        print("\n❌ No models trained successfully!")
        return None, None
    
    # Combine results
    all_results_df = pd.concat(all_results, ignore_index=True)
    
    # 5. OPTIMIZE BEST MODEL (optional)
    print("\n\n5. MODEL OPTIMIZATION")
    print("-" * 40)
    
    best_xgb = None
    enhanced_features = None
    
    # Use our_enhanced features if available, otherwise paper_16
    if 'our_enhanced' in trained_feature_sets:
        enhanced_features = feature_sets['our_enhanced']
    elif 'paper_16' in trained_feature_sets:
        enhanced_features = feature_sets['paper_16']
    
    if enhanced_features and not args.no_optimize:
        try:
            X_train_scaled, imputer, scaler = feature_engineer.prepare_features(train_df, enhanced_features)
            X_val_scaled = scaler.transform(imputer.transform(val_df[enhanced_features]))
            
            # Optimize XGBoost
            best_xgb = model_trainer.optimize_xgboost(
                X_train_scaled, X_val_scaled,
                train_df['target'].values, val_df['target'].values
            )
            
            # Save the optimized model
            joblib.dump(best_xgb, f'models/xgb_optimized_{timestamp}.pkl')
            print(f"✓ Optimized XGBoost saved: models/xgb_optimized_{timestamp}.pkl")
            
        except Exception as e:
            print(f"⚠️  Optimization failed: {str(e)[:80]}...")
            print("   Using best model from comparison...")
    else:
        print("⚠️  Skipping optimization (disabled or no features available)")
    
    # If no optimized model, use best from comparison
    if best_xgb is None:
        try:
            best_row = all_results_df.loc[all_results_df['auc_roc'].idxmax()]
            best_model_name = best_row['model_name']
            best_feature_set = best_row['feature_set']
            best_model_key = f"{best_model_name}_{best_feature_set}"
            best_xgb = model_trainer.models.get(best_model_key)
            
            if best_xgb:
                joblib.dump(best_xgb, f'models/xgb_best_{timestamp}.pkl')
                print(f"✓ Best model saved: models/xgb_best_{timestamp}.pkl")
            else:
                print("⚠️  Could not retrieve best model")
        except:
            print("⚠️  Could not determine best model")
    
    # 6. BUSINESS ANALYSIS
    print("\n\n6. BUSINESS IMPACT ANALYSIS")
    print("-" * 40)
    
    best_threshold = 0.5
    best_metrics = {}
    profit_df = pd.DataFrame()
    
    if best_xgb is not None and enhanced_features:
        try:
            X_test_scaled = scaler.transform(imputer.transform(test_df[enhanced_features]))
            y_test = test_df['target'].values
            
            # Find optimal threshold
            best_threshold, best_metrics, profit_df = model_trainer.business_profit_analysis(
                best_xgb, X_test_scaled, y_test
            )
            
            print(f"✓ Optimal threshold: {best_threshold:.3f}")
            print(f"✓ Maximum profit: ${best_metrics.get('profit', 0):,.0f}")
            
        except Exception as e:
            print(f"⚠️  Business analysis failed: {str(e)[:80]}...")
    else:
        print("⚠️  Cannot perform business analysis")
    
    # 7. SAVE RESULTS
    print("\n\n7. SAVING RESULTS")
    print("-" * 40)
    
    try:
        # Save model comparison
        all_results_df.to_csv(f'results/model_comparison_{timestamp}.csv', index=False)
        print(f"✓ Model comparison: results/model_comparison_{timestamp}.csv")
        
        # Save profit analysis if available
        if not profit_df.empty:
            profit_df.to_csv(f'results/profit_analysis_{timestamp}.csv', index=False)
            print(f"✓ Profit analysis: results/profit_analysis_{timestamp}.csv")
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'training_mode': 'full' if CONFIG['data_settings']['sample_size'] is None else 'sampled',
            'sample_size': CONFIG['data_settings']['sample_size'],
            'best_threshold': float(best_threshold),
            'best_auc_roc': float(all_results_df['auc_roc'].max()),
            'best_model': all_results_df.loc[all_results_df['auc_roc'].idxmax(), 'model_name'],
            'best_features': all_results_df.loc[all_results_df['auc_roc'].idxmax(), 'feature_set'],
            'improvement': {
                'paper_16_avg_auc': float(all_results_df[all_results_df['feature_set'] == 'paper_16']['auc_roc'].mean() if 'paper_16' in all_results_df['feature_set'].unique() else 0),
                'our_enhanced_avg_auc': float(all_results_df[all_results_df['feature_set'] == 'our_enhanced']['auc_roc'].mean() if 'our_enhanced' in all_results_df['feature_set'].unique() else 0)
            }
        }
        
        with open(f'results/summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary: results/summary_{timestamp}.json")
        
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")
    
    # 8. CREATE VISUALIZATIONS (optional)
    if not args.no_viz:
        print("\n\n8. CREATING VISUALIZATIONS")
        print("-" * 40)
        
        try:
            create_visualizations(all_results_df, profit_df, best_threshold, timestamp)
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
    
    # 9. FINAL SUMMARY
    print("\n\n" + "=" * 80)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 80)
    
    # Show key results
    print("\n📊 KEY RESULTS:")
    print("-" * 40)
    
    # Best overall model
    best_idx = all_results_df['auc_roc'].idxmax()
    best_result = all_results_df.loc[best_idx]
    print(f"🏆 BEST MODEL: {best_result['model_name']}")
    print(f"   Features: {best_result['feature_set']}")
    print(f"   AUC-ROC: {best_result['auc_roc']:.4f}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    
    # Improvement analysis
    if 'paper_16' in trained_feature_sets and 'our_enhanced' in trained_feature_sets:
        paper_avg = all_results_df[all_results_df['feature_set'] == 'paper_16']['auc_roc'].mean()
        our_avg = all_results_df[all_results_df['feature_set'] == 'our_enhanced']['auc_roc'].mean()
        improvement = our_avg - paper_avg
        
        print(f"\n📈 IMPROVEMENT ANALYSIS:")
        print(f"   Paper's features (avg): {paper_avg:.4f}")
        print(f"   Our features (avg): {our_avg:.4f}")
        print(f"   Improvement: +{improvement:.4f}")
    
    # Business impact
    if best_metrics:
        print(f"\n💰 BUSINESS IMPACT:")
        print(f"   Optimal threshold: {best_threshold:.3f}")
        print(f"   Maximum profit: ${best_metrics.get('profit', 0):,.0f}")
        print(f"   Accuracy at threshold: {best_metrics.get('accuracy', 0):.3f}")
    
    print(f"\n💾 OUTPUTS SAVED:")
    print(f"   Results: results/model_comparison_{timestamp}.csv")
    print(f"   Summary: results/summary_{timestamp}.json")
    print(f"   Data info: results/data_info_{timestamp}.json")
    
    if best_xgb:
        print(f"   Model: models/xgb_{'optimized' if not args.no_optimize else 'best'}_{timestamp}.pkl")
    
    print("\n" + "=" * 80)
    
    return best_xgb, best_threshold

def create_visualizations(results_df, profit_df, best_threshold, timestamp):
    """Create visualization plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Model Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = results_df['model_name'].unique()
    feature_sets = results_df['feature_set'].unique()
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, fs in enumerate(feature_sets):
        fs_data = results_df[results_df['feature_set'] == fs]
        auc_values = []
        for model in models:
            model_data = fs_data[fs_data['model_name'] == model]
            if len(model_data) > 0:
                auc_values.append(model_data['auc_roc'].values[0])
            else:
                auc_values.append(0)
        
        offset = -width/2 if i == 0 else width/2
        bars = ax.bar(x + offset, auc_values, width, label=fs, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, auc_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Performance by Feature Set')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/model_performance_{timestamp}.png', dpi=150)
    plt.close()
    print(f"✓ Model performance plot saved")
    
    # 2. Profit Analysis (if available)
    if not profit_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        ax2.plot(profit_df['threshold'], profit_df['profit']/1000, 'b-', linewidth=2)
        ax2.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'Optimal: {best_threshold:.3f}')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Profit (Thousands $)')
        ax2.set_title('Business Profit vs. Classification Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/profit_curve_{timestamp}.png', dpi=150)
        plt.close()
        print(f"✓ Profit curve saved")
    
    # 3. Training Time Comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    for fs in feature_sets:
        subset = results_df[results_df['feature_set'] == fs]
        ax3.bar(subset['model_name'], subset['train_time'], alpha=0.7, label=fs)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/training_time_{timestamp}.png', dpi=150)
    plt.close()
    print(f"✓ Training time plot saved")

if __name__ == "__main__":
    try:
        # Run the main pipeline
        best_model, optimal_threshold = main()
        
        # Show usage tips
        print("\n💡 USAGE TIPS:")
        print("-" * 40)
        print("For different training modes:")
        print("  python train.py --quick          # Quick test (50k samples)")
        print("  python train.py --full           # Full training (all data)")
        print("  python train.py --sample 100000  # Custom sample size")
        print("  python train.py --no-optimize    # Skip Bayesian optimization")
        print("  python train.py --no-viz         # Skip visualizations")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except MemoryError:
        print("\n\n❌ MEMORY ERROR!")
        print("The dataset is too large for available RAM.")
        print("Try: python train.py --quick or python train.py --sample 50000")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)