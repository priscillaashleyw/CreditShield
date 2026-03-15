"""
Main training script - Train TabTransformer on real Lending Club data
"""
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from load_data import DataLoader
from train_tab_transformer import TabTransformerTrainer

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent.parent / 'config' / 'tab_transformer_config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuration loaded from {config_path}")
    return config

def check_for_local_data():
    """Check if data exists locally and return path"""
    data_dir = Path(__file__).parent / 'data'
    
    if not data_dir.exists():
        return None
    
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        return None
    
    # Sort by size, largest first
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    largest = csvs[0]
    
    size_mb = largest.stat().st_size / (1024 * 1024)
    print(f"✓ Found local data: {largest.name} ({size_mb:.2f} MB)")
    
    return str(largest)

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("TabTransformer Training for Credit Risk Analysis")
    print("=" * 70)
    
    # Check for local data FIRST before loading config
    print("\n[1/5] Checking for data...")
    local_data = check_for_local_data()
    
    if not local_data:
        print("\n✗ No data found!")
        print("\nPlease create sample data first:")
        print("  python create_sample_data.py")
        print("\nOr download real data from Kaggle.")
        sys.exit(1)
    
    # Load configuration
    print("\n[2/5] Loading configuration...")
    config = load_config()
    
    # Override config to use local data
    config['paths']['raw_data'] = local_data
    print(f"✓ Using local data: {local_data}")
    
    # Load and preprocess data
    print("\n[3/5] Loading and preprocessing data...")
    try:
        loader = DataLoader(config)
        df = loader.load_and_filter_data()
        print(f"✓ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        # Show data info
        print(f"\nDataset columns:")
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'int16', 'int8']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"  Numerical ({len(numeric_cols)}): {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}")
        print(f"  Categorical ({len(cat_cols)}): {cat_cols}")
        
        # Define target variable
        df_clean = loader.define_target(df, strategy='business')
        print(f"✓ Target defined: {len(df_clean):,} loans with outcomes")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Split data
    print("\n[4/5] Splitting data (stratified by target)...")
    try:
        X_train, X_test, y_train, y_test = loader.random_split(
            df_clean, 
            test_size=0.2, 
            random_state=42
        )
        print(f"✓ Training: {len(X_train):,} samples")
        print(f"✓ Testing: {len(X_test):,} samples")
        print(f"✓ Training default rate: {y_train.mean():.2%}")
        print(f"✓ Testing default rate: {y_test.mean():.2%}")
        
    except Exception as e:
        print(f"✗ Data splitting failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Train model
    print("\n[5/5] Training TabTransformer...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Using device: {device}")
        
        # Flatten training config into main config for trainer
        training_config = config.get('training', {})
        config.update(training_config)
        
        trainer = TabTransformerTrainer(config, device=device)
        model, metrics = trainer.train(X_train, X_test, y_train, y_test)
        
        # Print final results
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\n📊 Final Evaluation Metrics on TEST SET:")
        print("-" * 40)
        print(f"  AUC-ROC:    {metrics['auc']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1-Score:   {metrics['f1']:.4f}")
        print("-" * 40)
        
        # Save metrics
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = results_dir / 'metrics.yaml'
        with open(metrics_path, 'w') as f:
            yaml.dump({k: float(v) for k, v in metrics.items()}, f)
        print(f"\n✓ Metrics saved to {metrics_path}")
        
        # Save model info
        model_info = {
            'model_type': 'TabTransformer',
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'device': device,
            'training_samples': int(len(X_train)),
            'testing_samples': int(len(X_test)),
            'default_rate_train': float(y_train.mean()),
            'default_rate_test': float(y_test.mean()),
            'metrics': {k: float(v) for k, v in metrics.items()}
        }
        
        info_path = results_dir / 'model_info.yaml'
        with open(info_path, 'w') as f:
            yaml.dump(model_info, f)
        print(f"✓ Model info saved to {info_path}")
        
        print("\n" + "=" * 70)
        return 0
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
