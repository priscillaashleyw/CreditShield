"""
Integration test - Test TabTransformer with actual data pipeline
"""
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from load_data import DataLoader
from train_tab_transformer import TabTransformerTrainer

def create_dummy_config():
    """Create a minimal config for testing"""
    return {
        'paths': {
            'raw_data': 'training/data/dummy_lending_club.csv'
        },
        'data_settings': {
            'essential_columns': ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'term', 'loan_status', 'issue_d'],
            'years': [2013, 2014],
            'incomplete_statuses': ['Current'],
            'leaked_columns': []
        },
        'target_settings': {
            'business': {
                'default_statuses': ['Charged Off', 'Default'],
                'good_statuses': ['Fully Paid']
            }
        },
        'model': {
            'embedding_dim': 16,
            'depth': 2,
            'heads': 4,
            'dim_head': 32,
            'mlp_dim': 128,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'epochs': 3,
            'learning_rate': 0.001,
            'early_stopping_patience': 5
        },
        'batch_size': 32,
        'epochs': 3,
        'learning_rate': 0.001,
        'early_stopping_patience': 5
    }

def create_dummy_data(path, n_samples=500):
    """Create dummy lending club data for testing"""
    np.random.seed(42)
    
    data = {
        'loan_amnt': np.random.uniform(1000, 35000, n_samples),
        'int_rate': np.random.uniform(5, 30, n_samples),
        'annual_inc': np.random.uniform(20000, 200000, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'term': np.random.choice(['36 months', '60 months'], n_samples),
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off', 'Default', 'Current'], n_samples),
        'issue_d': ['Jan-2013' if i < n_samples//2 else 'Jan-2014' for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✓ Created dummy data at {path} with {n_samples} samples")
    return df

def test_integration():
    """Test full pipeline: data loading -> preprocessing -> model training"""
    print("=" * 60)
    print("INTEGRATION TEST: Data Pipeline -> TabTransformer")
    print("=" * 60)
    
    # Setup
    config = create_dummy_config()
    data_path = Path(config['paths']['raw_data'])
    
    # Create dummy data
    print("\n[1/4] Creating dummy data...")
    create_dummy_data(data_path)
    
    # Test data loading
    print("\n[2/4] Testing data loading and preprocessing...")
    try:
        loader = DataLoader(config)
        df = loader.load_and_filter_data()
        print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Define target
        df_clean = loader.define_target(df, strategy='business')
        print(f"✓ Target defined: {len(df_clean)} loans with outcomes")
        
        # Split data
        X_train, X_test, y_train, y_test = loader.random_split(df_clean, test_size=0.2)
        print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test model training
    print("\n[3/4] Testing TabTransformer training...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = TabTransformerTrainer(config, device=device)
        model, metrics = trainer.train(X_train, X_test, y_train, y_test)
        print(f"✓ Training completed successfully")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    print("\n[4/4] Cleaning up...")
    try:
        data_path.unlink()
        if Path('best_tab_transformer.pth').exists():
            Path('best_tab_transformer.pth').unlink()
        print("✓ Cleanup completed")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")
    
    print("\n" + "=" * 60)
    print("✓ INTEGRATION TEST PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
