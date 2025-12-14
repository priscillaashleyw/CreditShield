# deployment/predictor_fixed.py
import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path
import json

class CreditRiskPredictorFixed:
    """Predictor using your enhanced feature set"""
    
    def __init__(self, model_dir="model_artifacts"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.imputer = None
        self.optimal_threshold = 0.28
        
        # Load the feature list we saved
        self.feature_list = self._load_feature_list()
        print(f"📋 Using {len(self.feature_list)} enhanced features")
        
        self.load_artifacts()
    
    def _load_feature_list(self):
        """Load or create feature list"""
        feature_file = self.model_dir / "training_features.json"
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                data = json.load(f)
                return data.get('enhanced_features', self._default_features())
        else:
            return self._default_features()
    
    def _default_features(self):
        """Default enhanced feature set based on your training"""
        return [
            # Basic features
            'loan_amnt', 'int_rate', 'annual_inc', 'dti',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'total_acc',
            
            # Engineered numerical features
            'grade_numeric', 'emp_length_numeric', 'revol_util_decimal',
            'loan_to_income', 'int_rate_times_loan',
            'has_delinquencies', 'subprime_high_dti',
            
            # Paper's additional features (simplified)
            'years_since_earliest_cr', 'credit_utilization_ratio'
        ]
    
    def load_artifacts(self):
        """Load model, scaler, and imputer"""
        try:
            # Find the latest model (look for enhanced features model)
            model_files = list(self.model_dir.glob("*xgb*.pkl"))
            scaler_files = list(self.model_dir.glob("*scaler*.pkl"))
            imputer_files = list(self.model_dir.glob("*imputer*.pkl"))
            
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            # Load the first available
            self.model = joblib.load(model_files[0])
            print(f"✅ Loaded model: {model_files[0].name}")
            
            if scaler_files:
                self.scaler = joblib.load(scaler_files[0])
                print(f"✅ Loaded scaler: {scaler_files[0].name}")
            
            if imputer_files:
                self.imputer = joblib.load(imputer_files[0])
                print(f"✅ Loaded imputer: {imputer_files[0].name}")
            
            # Try to determine actual number of features
            if hasattr(self.model, 'n_features_in_'):
                print(f"📊 Model expects {self.model.n_features_in_} features")
            elif hasattr(self.model, 'feature_importances_'):
                print(f"📊 Model has {len(self.model.feature_importances_)} feature importances")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise
    
    def _engineer_features(self, df):
        """Create all enhanced features from basic input"""
        # Grade to numeric
        if 'grade' in df.columns:
            grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            df['grade_numeric'] = df['grade'].map(grade_map)
        
        # Employment length
        if 'emp_length' in df.columns:
            df['emp_length_numeric'] = df['emp_length'].apply(self._convert_emp_length)
        
        # Credit utilization
        if 'revol_util' in df.columns:
            df['revol_util_decimal'] = df['revol_util'].astype(str).str.replace('%', '').astype(float) / 100
        
        # Financial ratios
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
            df['int_rate_times_loan'] = df['int_rate'] * df['loan_amnt'] / 1000
        
        # Credit flags
        if 'delinq_2yrs' in df.columns:
            df['has_delinquencies'] = (df['delinq_2yrs'] > 0).astype(int)
        
        # Subprime indicator
        if 'grade_numeric' in df.columns and 'dti' in df.columns:
            df['subprime_high_dti'] = ((df['grade_numeric'] >= 4) & (df['dti'] > 20)).astype(int)
        
        # Simplified paper features
        df['years_since_earliest_cr'] = 10  # Default
        df['credit_utilization_ratio'] = df['revol_util_decimal'] if 'revol_util_decimal' in df.columns else 0.5
        
        return df
    
    def _convert_emp_length(self, val):
        """Convert employment length string to numeric"""
        if pd.isna(val):
            return np.nan
        val = str(val).lower()
        if '10+' in val:
            return 10
        elif '< 1' in val:
            return 0
        else:
            numbers = re.findall(r'\d+', val)
            return float(numbers[0]) if numbers else np.nan
    
    def preprocess_input(self, input_dict):
        """Convert raw input to model-ready features"""
        df = pd.DataFrame([input_dict])
        
        # Engineer all features
        df = self._engineer_features(df)
        
        # Create empty dataframe with all expected features
        processed_df = pd.DataFrame(columns=self.feature_list)
        
        # Fill with zeros first
        for feature in self.feature_list:
            processed_df[feature] = 0
        
        # Copy available values
        for feature in self.feature_list:
            if feature in df.columns:
                processed_df[feature] = df[feature].values
        
        # Handle missing values
        if self.imputer:
            processed_df = pd.DataFrame(
                self.imputer.transform(processed_df),
                columns=self.feature_list
            )
        
        # Scale features
        if self.scaler:
            processed_df = pd.DataFrame(
                self.scaler.transform(processed_df),
                columns=self.feature_list
            )
        
        return processed_df.values
    
    def predict(self, input_dict):
        """Make prediction"""
        try:
            # Preprocess
            features = self.preprocess_input(input_dict)
            
            # Debug info
            print(f"🔧 Processed features shape: {features.shape}")
            
            # Predict
            default_prob = self.model.predict_proba(features)[0, 1]
            
            # Decision
            decision = "APPROVE" if default_prob < self.optimal_threshold else "REJECT"
            
            return {
                'success': True,
                'default_probability': float(default_prob),
                'decision': decision,
                'risk_level': self._get_risk_level(default_prob),
                'confidence': self._get_confidence(default_prob),
                'optimal_threshold': self.optimal_threshold,
                'explanation': f"Default probability: {default_prob:.1%} (threshold: {self.optimal_threshold:.1%})"
            }
            
        except Exception as e:
            import traceback
            print(f"❌ Prediction error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'decision': 'ERROR'
            }
    
    def _get_risk_level(self, prob):
        if prob < 0.2: return "LOW"
        elif prob < 0.4: return "MEDIUM"
        elif prob < 0.6: return "HIGH"
        else: return "VERY HIGH"
    
    def _get_confidence(self, prob):
        distance = abs(prob - self.optimal_threshold)
        return max(0.5, 1.0 - distance * 2)

# Test
if __name__ == "__main__":
    print("🧪 Testing CreditRiskPredictorFixed...")
    
    # Copy model files to deployment folder if needed
    import shutil
    from pathlib import Path
    
    # Ensure deployment folder has model files
    deployment_dir = Path("../deployment/model_artifacts")
    training_dir = Path("models")
    
    if not list(deployment_dir.glob("*.pkl")):
        print("📦 Copying model files to deployment folder...")
        deployment_dir.mkdir(exist_ok=True)
        
        # Copy latest model files
        for pattern in ["xgb_*.pkl", "scaler_*.pkl", "imputer_*.pkl"]:
            files = list(training_dir.glob(pattern))
            if files:
                latest = max(files, key=lambda x: x.stat().st_mtime)
                shutil.copy2(latest, deployment_dir / latest.name)
                print(f"  Copied: {latest.name}")
    
    # Create predictor
    predictor = CreditRiskPredictorFixed("model_artifacts")
    
    # Test prediction
    test_loan = {
        'loan_amnt': 15000,
        'int_rate': 12.5,
        'grade': 'C',
        'emp_length': '5 years',
        'annual_inc': 75000,
        'dti': 18.5,
        'revol_util': '45%',
        'delinq_2yrs': 0,
        'inq_last_6mths': 2,
        'open_acc': 8,
        'total_acc': 25
    }
    
    print("\n📊 Making test prediction...")
    result = predictor.predict(test_loan)
    
    print("\n📈 Prediction Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")