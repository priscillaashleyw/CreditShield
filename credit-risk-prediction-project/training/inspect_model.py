# training/inspect_model.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

def inspect_all_models():
    """Inspect all trained models to understand their features"""
    
    print("🔍 Inspecting trained models...")
    print("=" * 60)
    
    # Find all model files
    model_dir = Path("models")
    model_files = list(model_dir.glob("*.pkl"))
    
    if not model_files:
        print("No model files found in models/ directory")
        return
    
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        print(f"  - {file.name}")
    
    print("\n" + "=" * 60)
    
    # Try to load each model and inspect it
    for model_file in model_files:
        print(f"\n📊 Inspecting: {model_file.name}")
        
        try:
            # Try to load as XGBoost model
            model = joblib.load(model_file)
            
            # Check model type
            print(f"  Model type: {type(model).__name__}")
            
            # Check for feature names
            if hasattr(model, 'get_booster'):
                # XGBoost model
                booster = model.get_booster()
                feature_names = booster.feature_names
                if feature_names:
                    print(f"  Feature names available via booster: {len(feature_names)} features")
                    print(f"  First 10: {feature_names[:10]}")
                else:
                    print("  No feature names in booster")
            
            elif hasattr(model, 'feature_names_in_'):
                # Scikit-learn model
                print(f"  Feature names: {len(model.feature_names_in_)} features")
                print(f"  First 10: {model.feature_names_in_[:10]}")
            
            elif hasattr(model, 'feature_importances_'):
                # Has feature importances
                print(f"  Has feature_importances_: {len(model.feature_importances_)} features")
            
            elif hasattr(model, 'coef_'):
                # Linear model
                if len(model.coef_.shape) > 1:
                    print(f"  Coefficients shape: {model.coef_.shape}")
                else:
                    print(f"  Coefficients: {len(model.coef_)} features")
            
            # Check for other attributes
            attrs = [attr for attr in dir(model) if not attr.startswith('_')]
            print(f"  Available attributes: {len(attrs)}")
            print(f"  Some attributes: {attrs[:10]}...")
            
            # Try to get number of features
            if hasattr(model, 'n_features_in_'):
                print(f"  n_features_in_: {model.n_features_in_}")
            
        except Exception as e:
            print(f"  ❌ Error loading {model_file.name}: {e}")
    
    print("\n" + "=" * 60)
    
    # Also check scalers and imputers
    print("\n🔍 Inspecting scalers and imputers...")
    
    scaler_files = list(model_dir.glob("scaler_*.pkl"))
    imputer_files = list(model_dir.glob("imputer_*.pkl"))
    
    print(f"Scalers: {len(scaler_files)} files")
    print(f"Imputers: {len(imputer_files)} files")
    
    if scaler_files:
        scaler = joblib.load(scaler_files[0])
        print(f"\nFirst scaler shape: {scaler.mean_.shape if hasattr(scaler, 'mean_') else 'Unknown'}")
    
    if imputer_files:
        imputer = joblib.load(imputer_files[0])
        print(f"First imputer shape: {imputer.statistics_.shape if hasattr(imputer, 'statistics_') else 'Unknown'}")

if __name__ == "__main__":
    inspect_all_models()