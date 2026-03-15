#!/usr/bin/env python3
"""
Run TabTransformer training on Lending Club data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
from src.load_data import DataLoader
from src.train_tab_transformer import TabTransformerTrainer


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "tab_transformer_config.yaml"
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("TabTransformer Training for Credit Risk Prediction")
    print("Output: Probability (0-1)")
    print("=" * 60)
    
    # Load and prepare data
    print("\n[1/3] Loading data...")
    data_loader = DataLoader(config)
    df = data_loader.load_and_filter_data()
    df = data_loader.define_target(df, strategy='business')
    
    # Split data
    print("\n[2/3] Splitting data...")
    X_train, X_test, y_train, y_test = data_loader.random_split(df, test_size=0.2)
    
    # Train TabTransformer
    print("\n[3/3] Training TabTransformer...")
    trainer = TabTransformerTrainer(config)
    model, metrics = trainer.train(X_train, X_test, y_train, y_test)
    
    # Save metrics to metrics.yaml
    results_dir = Path(__file__).parent / "results"
    metrics_path = results_dir / "metrics.yaml"
    
    metrics_to_save = {
        'model_type': 'TabTransformer',
        'output_type': 'probability (0-1)',
        'optimal_threshold': float(metrics['threshold']),
        'auc': float(metrics['auc']),
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'mae': float(metrics['mae']),
        'true_positives': int(metrics['true_positives']),
        'true_negatives': int(metrics['true_negatives']),
        'false_positives': int(metrics['false_positives']),
        'false_negatives': int(metrics['false_negatives']),
    }
    
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics_to_save, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Training complete!")
    print(f"Model saved to: {results_dir / 'best_tab_transformer.pth'}")
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    main()
