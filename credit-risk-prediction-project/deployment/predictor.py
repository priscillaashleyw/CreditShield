# deployment/predictor.py
import joblib
import numpy as np
import pandas as pd
import re
from pathlib import Path
import json

class CreditRiskPredictor:
    """Predictor using your actual trained model features"""
    
    def __init__(self, model_dir="model_artifacts"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.imputer = None
        self.optimal_threshold = 0.28
        
        # Load the ACTUAL feature list from your JSON
        self.feature_list = self._load_actual_features()
        print(f"📋 Using {len(self.feature_list)} ACTUAL features")
        
        # Extract base features needed from user input
        self.base_features_needed = self._extract_base_features()
        print(f"📋 Expecting {len(self.base_features_needed)} base input features")
        
        self.load_artifacts()
    
    def _load_actual_features(self):
        """Load the actual features used in training"""
        feature_file = self.model_dir / "training_features.json"
        if not feature_file.exists():
            print(f"⚠️ {feature_file} not found")
            return []
        
        with open(feature_file, 'r') as f:
            data = json.load(f)
        
        # Your JSON has 'feature_names' key
        if 'feature_names' in data:
            features = data['feature_names']
            if isinstance(features, list):
                return features
        elif 'enhanced_features' in data:
            features = data['enhanced_features']
            if isinstance(features, list):
                return features
        
        print(f"❌ Could not find feature list in JSON. Keys: {list(data.keys())}")
        return []
    
    def _extract_base_features(self):
        """Extract base features from one-hot encoded feature list"""
        if not self.feature_list:
            return []
            
        base_features = set()
        for feature in self.feature_list:
            # Handle one-hot encoded features
            if feature.startswith('addr_state_'):
                base_features.add('addr_state')
            elif feature.startswith('home_ownership_'):
                base_features.add('home_ownership')
            elif feature.startswith('purpose_'):
                base_features.add('purpose')
            elif feature.startswith('verification_status_'):
                base_features.add('verification_status')
            elif feature.startswith('title_has_'):
                # These are title-based engineered features
                base_features.add('title')
            elif '_' in feature and not feature.replace('_', '').isnumeric():
                # Other potential categoricals
                parts = feature.split('_')
                if len(parts) > 1:
                    base_features.add(parts[0])
            else:
                # Regular feature
                base_features.add(feature)
        
        # Filter out features that don't make sense as user inputs
        user_input_features = []
        for feature in base_features:
            if feature not in ['purpose_debt_consolidation', 'verification_status_Verified', 
                              'verification_status_Source', 'title_has_car', 'title_has_medical',
                              'title_has_credit', 'title_has_home', 'title_has_consolidation',
                              'title_has_debt', 'title_has_card'] and not any(feature + '_' in f for f in self.feature_list):
                user_input_features.append(feature)
        
        return user_input_features
    
    def load_artifacts(self):
        """Load model, scaler, and imputer"""
        try:
            # Find the latest model
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
            
            # Verify feature count
            if hasattr(self.model, 'n_features_in_'):
                print(f"📊 Model expects {self.model.n_features_in_} features")
                print(f"📊 We have {len(self.feature_list)} features in our list")
                
                if self.model.n_features_in_ != len(self.feature_list):
                    print("⚠️ WARNING: Feature count mismatch!")
            
        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise
    
    def _engineer_features(self, df):
        """Create all features including one-hot encoded"""
        if not self.feature_list:
            raise ValueError("No feature list available!")
            
        # First, ensure we have all base features (fill missing with defaults)
        for feature in self.base_features_needed:
            if feature not in df.columns:
                # Set appropriate defaults based on feature type
                if feature in ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'total_acc', 
                              'revol_bal', 'total_bc_limit', 'total_bal_ex_mort', 'avg_cur_bal',
                              'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
                              'mths_since_recent_bc', 'mths_since_recent_inq', 'last_fico_range_low',
                              'last_fico_range_high', 'years_since_earliest_cr']:
                    df[feature] = 0  # Numerical defaults
                elif feature in ['addr_state', 'home_ownership', 'purpose', 'verification_status', 'title']:
                    df[feature] = 'unknown'  # Categorical defaults
                elif feature in ['grade_numeric', 'emp_length_numeric', 'revol_util_decimal',
                                'loan_to_income', 'int_rate_times_loan', 'subprime_high_dti',
                                'pct_tl_nvr_dlq', 'title_length', 'title_word_count']:
                    df[feature] = 0  # Engineered feature defaults
                elif feature in ['delinq_2yrs', 'inq_last_6mths', 'open_acc', 'has_delinq_history']:
                    df[feature] = 0  # Credit history defaults
                else:
                    df[feature] = 0
        
        # Convert categorical to one-hot
        df = self._create_one_hot_features(df)
        
        # Engineered features
        df = self._create_engineered_features(df)
        
        return df
    
    def _create_one_hot_features(self, df):
        """Create one-hot encoded features from categorical variables"""
        if not self.feature_list:
            return df
            
        for feature in self.feature_list:
            # Handle different categorical encodings
            if feature.startswith('addr_state_'):
                state_code = feature.replace('addr_state_', '')
                if 'addr_state' in df.columns:
                    df[feature] = (df['addr_state'].astype(str).str.upper() == state_code).astype(int)
                else:
                    df[feature] = 0
            
            elif feature.startswith('home_ownership_'):
                ownership_type = feature.replace('home_ownership_', '')
                if 'home_ownership' in df.columns:
                    df[feature] = (df['home_ownership'].astype(str).str.upper() == ownership_type).astype(int)
                else:
                    df[feature] = 0
                    
            elif feature.startswith('purpose_'):
                purpose_type = feature.replace('purpose_', '')
                if 'purpose' in df.columns:
                    df[feature] = (df['purpose'].astype(str).str.lower().replace(' ', '_') == purpose_type).astype(int)
                else:
                    df[feature] = 0
                    
            elif feature.startswith('verification_status_'):
                status_type = feature.replace('verification_status_', '')
                if 'verification_status' in df.columns:
                    df[feature] = (df['verification_status'].astype(str).str.replace(' ', '_') == status_type).astype(int)
                else:
                    df[feature] = 0
                    
            elif feature.startswith('title_has_'):
                # These are title-based engineered features
                keyword = feature.replace('title_has_', '')
                if 'title' in df.columns:
                    title_str = str(df['title'].iloc[0]).lower() if len(df) > 0 else ''
                    df[feature] = 1 if keyword in title_str else 0
                else:
                    df[feature] = 0
        
        return df
    
    def _create_engineered_features(self, df):
        """Create engineered features"""
        # Grade to numeric (if grade is provided)
        if 'grade' in df.columns:
            grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            df['grade_numeric'] = df['grade'].map(grade_map).fillna(4)
        
        # Employment length to numeric
        if 'emp_length' in df.columns:
            df['emp_length_numeric'] = df['emp_length'].apply(self._convert_emp_length)
        
        # Credit utilization to decimal
        if 'revol_util' in df.columns:
            df['revol_util_decimal'] = df['revol_util'].astype(str).str.replace('%', '', regex=False).astype(float) / 100
        
        # Financial ratios
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'].replace(0, 1) + 1)
        
        if 'int_rate' in df.columns and 'loan_amnt' in df.columns:
            df['int_rate_times_loan'] = df['int_rate'] * df['loan_amnt'] / 1000
        
        # Credit flags
        if 'delinq_2yrs' in df.columns:
            df['has_delinq_history'] = (df['delinq_2yrs'] > 0).astype(int)
        
        # Subprime indicator
        if 'grade_numeric' in df.columns and 'dti' in df.columns:
            df['subprime_high_dti'] = ((df['grade_numeric'] >= 4) & (df['dti'] > 20)).astype(int)
        
        # Title-based features
        if 'title' in df.columns:
            title_str = str(df['title'].iloc[0]).lower() if len(df) > 0 else ''
            df['title_length'] = len(title_str)
            df['title_word_count'] = len(title_str.split())
        
        # Years since earliest credit line (simplified)
        if 'years_since_earliest_cr' not in df.columns:
            df['years_since_earliest_cr'] = 10  # Default value
        
        # Set defaults for any missing engineered features
        for feature in self.feature_list:
            if feature not in df.columns and not feature.startswith(('addr_state_', 'home_ownership_', 
                                                                    'purpose_', 'verification_status_', 'title_has_')):
                # Default values based on feature type
                if 'fico' in feature.lower():
                    df[feature] = 700  # Average FICO score
                elif any(x in feature for x in ['rate', 'util', 'pct', 'ratio']):
                    df[feature] = 0.5  # Percentage default
                elif any(x in feature for x in ['loan', 'amt', 'bal', 'limit', 'inc']):
                    df[feature] = 0  # Monetary default
                elif any(x in feature for x in ['month', 'mo', 'mth', 'year']):
                    df[feature] = 0  # Time default
                else:
                    df[feature] = 0
        
        return df
    
    def _convert_emp_length(self, val):
        """Convert employment length string to numeric"""
        if pd.isna(val):
            return 3.0  # Default
        val = str(val).lower()
        if '10+' in val:
            return 10.0
        elif '< 1' in val:
            return 0.5
        else:
            numbers = re.findall(r'\d+', val)
            return float(numbers[0]) if numbers else 3.0
    
    def preprocess_input(self, input_dict):
        """Convert raw input to model-ready features"""
        if not self.feature_list:
            raise ValueError("No feature list available!")
            
        df = pd.DataFrame([input_dict])
        
        # Engineer all features including one-hot
        df = self._engineer_features(df)
        
        # Ensure we have all features in correct order
        processed_df = pd.DataFrame(columns=self.feature_list)
        
        # Fill with available values, zeros for missing
        for feature in self.feature_list:
            if feature in df.columns:
                processed_df[feature] = df[feature].values
            else:
                processed_df[feature] = 0
        
        # Debug: Show we have the right number of features
        print(f"🔧 Created dataframe with {len(processed_df.columns)} features")
        
        # Handle missing values (imputer)
        if self.imputer is not None and not processed_df.empty:
            try:
                processed_df = pd.DataFrame(
                    self.imputer.transform(processed_df),
                    columns=self.feature_list
                )
            except Exception as e:
                print(f"⚠️ Imputer error: {e}")
        
        # Scale features
        if self.scaler is not None and not processed_df.empty:
            try:
                processed_df = pd.DataFrame(
                    self.scaler.transform(processed_df),
                    columns=self.feature_list
                )
            except Exception as e:
                print(f"⚠️ Scaler error: {e}")
        
        return processed_df.values
    
    def predict(self, input_dict):
        """Make prediction"""
        try:
            # Preprocess
            features = self.preprocess_input(input_dict)
            
            if features.size == 0:
                raise ValueError("No features generated!")
            
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

# Test with the exact features your model expects
if __name__ == "__main__":
    print("🧪 Testing CreditRiskPredictor...")
    print("=" * 60)
    
    # Create predictor
    predictor = CreditRiskPredictor("model_artifacts")
    
    if not predictor.feature_list:
        print("\n❌ Cannot proceed without features!")
    else:
        # Create a test input with ALL the features your model actually needs
        # Based on your JSON, here's what to provide:
        test_loan = {
            # Basic loan info
            'loan_amnt': 15000,
            'int_rate': 12.5,
            
            # Categorical features (will be one-hot encoded)
            'addr_state': 'CA',  # Will create addr_state_CA = 1
            'home_ownership': 'RENT',  # Will create home_ownership_RENT = 1
            'purpose': 'debt_consolidation',  # Will create purpose_debt_consolidation = 1
            'verification_status': 'Verified',  # Will create verification_status_Verified = 1
            
            # Title for title-based features
            'title': 'Debt consolidation loan for credit card payoff',
            
            # Credit features from your feature list
            'dti': 18.5,
            'annual_inc': 75000,
            'revol_util': '45%',
            'delinq_2yrs': 0,
            'inq_last_6mths': 2,
            'open_acc': 8,
            'total_acc': 25,
            'revol_bal': 5000,
            'total_bc_limit': 20000,
            'total_bal_ex_mort': 30000,
            'avg_cur_bal': 2500,
            'mo_sin_old_il_acct': 60,
            'mo_sin_old_rev_tl_op': 48,
            'mo_sin_rcnt_rev_tl_op': 12,
            'mths_since_recent_bc': 6,
            'mths_since_recent_inq': 3,
            'pct_tl_nvr_dlq': 0.95,
            'last_fico_range_low': 680,
            'last_fico_range_high': 684,
            
            # Additional features that might be needed
            'grade': 'C',
            'emp_length': '5 years',
            'years_since_earliest_cr': 10
        }
        
        print(f"\n📊 Making test prediction...")
        print(f"Using input with {len(test_loan)} fields")
        
        result = predictor.predict(test_loan)
        
        print("\n" + "=" * 60)
        print("📈 PREDICTION RESULTS:")
        print("=" * 60)
        for key, value in result.items():
            if key != 'explanation' or result['success']:
                print(f"{key:25}: {value}")