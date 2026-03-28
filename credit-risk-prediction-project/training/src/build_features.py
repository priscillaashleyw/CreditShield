"""
build_features.py - Feature engineering with paper's methodology + improvements
"""
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering following paper methodology with improvements"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def create_paper_features(self, df):
        """
        Create features according to paper's methodology:
        - Convert categorical features to numeric
        - Handle missing values as per paper
        - Create derived features
        """
        print("\nCreating features following paper methodology...")
        
        # 1. Convert employment length (ordinal encoding as per paper)
        df['emp_length_numeric'] = df['emp_length'].apply(self._emp_to_numeric)
        
        # 2. Convert revol_util from percentage to decimal
        df['revol_util_decimal'] = df['revol_util'].astype(str).str.replace('%', '').astype(float) / 100
        
        # 3. Convert earliest credit line to years from reference date (as per paper)
        reference_date = pd.Timestamp('2018-09-01')  # September 2018 as per paper
        df['earliest_cr_line_date'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
        df['years_since_earliest_cr'] = (reference_date - df['earliest_cr_line_date']).dt.days / 365.25
        
        # 4. Create missing value indicators as per paper
        # For features like 'mths_since_last_delinq' where NA means no delinquency
        if 'mths_since_last_delinq' in df.columns:
            df['has_delinq_history'] = df['mths_since_last_delinq'].notna().astype(int)
            df['mths_since_last_delinq_filled'] = df['mths_since_last_delinq'].fillna(-1)
        
        if 'mths_since_last_record' in df.columns:
            df['has_public_record'] = df['mths_since_last_record'].notna().astype(int)
            df['mths_since_last_record_filled'] = df['mths_since_last_record'].fillna(-1)
        
        # 5. One-hot encoding for categorical features without ordinal relationship
        categorical_cols = ['home_ownership', 'verification_status', 'purpose', 'addr_state']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        # 6. Create interaction terms (our improvement)
        df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
        df['int_rate_times_loan'] = df['int_rate'] * df['loan_amnt'] / 1000
        
        # 7. Create text-based features from loan title (NLP improvement)
        if 'title' in df.columns:
            df['title_length'] = df['title'].astype(str).str.len()
            df['title_word_count'] = df['title'].astype(str).str.split().str.len()
            # Simple keyword extraction
            keywords = ['debt', 'consolidation', 'credit', 'card', 'medical', 'home', 'car']
            for keyword in keywords:
                df[f'title_has_{keyword}'] = df['title'].astype(str).str.contains(keyword, case=False).astype(int)
        
        print(f"✓ Created {len([c for c in df.columns if 'target' not in c])} features total")

        # 8. Simple borrower profile risk heuristic
        # Higher score = higher risk

        score = np.zeros(len(df), dtype=float)

        # FICO (use midpoint if both columns exist)
        if {'last_fico_range_low', 'last_fico_range_high'}.issubset(df.columns):
            fico_mid = (df['last_fico_range_low'] + df['last_fico_range_high']) / 2

            score += np.where(fico_mid < 580, 25, 0)
            score += np.where((fico_mid >= 580) & (fico_mid < 640), 18, 0)
            score += np.where((fico_mid >= 640) & (fico_mid < 700), 10, 0)
            score += np.where((fico_mid >= 700) & (fico_mid < 750), 4, 0)
            score += np.where(fico_mid >= 750, 0, 0)

        # Debt-to-income
        if 'dti' in df.columns:
            score += np.where(df['dti'] > 30, 20, 0)
            score += np.where((df['dti'] > 20) & (df['dti'] <= 30), 10, 0)
            score += np.where((df['dti'] > 10) & (df['dti'] <= 20), 4, 0)

        # Delinquencies
        if 'delinq_2yrs' in df.columns:
            score += np.where(df['delinq_2yrs'] >= 3, 15, 0)
            score += np.where((df['delinq_2yrs'] >= 1) & (df['delinq_2yrs'] < 3), 8, 0)

        # Recent hard inquiries
        if 'inq_last_6mths' in df.columns:
            score += np.where(df['inq_last_6mths'] >= 4, 10, 0)
            score += np.where((df['inq_last_6mths'] >= 2) & (df['inq_last_6mths'] < 4), 5, 0)

        # Revolving utilization
        if 'revol_util_decimal' in df.columns:
            score += np.where(df['revol_util_decimal'] > 0.9, 12, 0)
            score += np.where((df['revol_util_decimal'] > 0.7) & (df['revol_util_decimal'] <= 0.9), 7, 0)
            score += np.where((df['revol_util_decimal'] > 0.5) & (df['revol_util_decimal'] <= 0.7), 3, 0)

        # Loan size relative to income
        if 'loan_to_income' in df.columns:
            score += np.where(df['loan_to_income'] > 0.5, 12, 0)
            score += np.where((df['loan_to_income'] > 0.3) & (df['loan_to_income'] <= 0.5), 7, 0)
            score += np.where((df['loan_to_income'] > 0.15) & (df['loan_to_income'] <= 0.3), 3, 0)

        # Employment length
        if 'emp_length_numeric' in df.columns:
            score += np.where(df['emp_length_numeric'].fillna(0) < 1, 6, 0)
            score += np.where(
                (df['emp_length_numeric'].fillna(0) >= 1) &
                (df['emp_length_numeric'].fillna(0) < 3), 3, 0
            )

        # Public record bankruptcies
        if 'pub_rec_bankruptcies' in df.columns:
            score += np.where(df['pub_rec_bankruptcies'] >= 1, 12, 0)

        # Income verification
        if 'verification_status' in df.columns:
            score += np.where(df['verification_status'].astype(str).str.lower() == 'not verified', 5, 0)

        # Cap and store
        df['risk_heuristic_score'] = np.clip(score, 0, 100)

        # Label buckets
        df['risk_heuristic_label'] = pd.cut(
            df['risk_heuristic_score'],
            bins=[-1, 20, 40, 60, 100],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )

        return df
    
    def create_feature_sets(self, df):
        """Create different feature sets for comparison"""
        
        # First, identify columns that are NOT features
        non_feature_columns = [
            'target', 'loan_status', 'issue_d', 'issue_date', 
            'issue_year', 'earliest_cr_line', 'earliest_cr_line_date',
            'emp_length', 'revol_util', 'title'
        ]
        
        # Get all available feature columns
        all_feature_columns = [col for col in df.columns if col not in non_feature_columns]
        
        # Paper's 16 features (from Table 5) - only keep those that exist
        paper_16_features = []
        for feature in self.config['feature_sets']['paper_16']:
            if feature in df.columns:
                paper_16_features.append(feature)
            else:
                # Try to find similar features
                similar = [col for col in df.columns if feature in col]
                if similar:
                    paper_16_features.extend(similar)
        
        # Remove duplicates
        paper_16_features = list(set(paper_16_features))
        
        # Our enhanced features (paper's 16 + our engineered features)
        our_features = paper_16_features.copy()
        
        # Add our engineered features that exist in the dataframe
        our_engineered_patterns = [
            'loan_to_income', 'int_rate_times_loan', 'subprime_high_dti',
            'emp_length_numeric', 'revol_util_decimal',
            'years_since_earliest_cr', 'has_delinq_history',
            'title_length', 'title_word_count', 
            'title_has_', 'risk_heuristic_score'
        ]
        
        for pattern in our_engineered_patterns:
            matching_features = [col for col in df.columns if pattern in col]
            our_features.extend(matching_features)
        
        # Add categorical dummies
        categorical_features = [col for col in df.columns if any(x in col for x in [
            'home_ownership_', 'verification_status_', 'purpose_', 'addr_state_'
        ])]
        our_features.extend(categorical_features)
        
        # Remove duplicates and non-feature columns
        our_features = list(set([f for f in our_features if f not in non_feature_columns]))
        
        print(f"\nFeature sets created:")
        print(f"  Paper's 16 features: {len(paper_16_features)}")
        print(f"  Our enhanced features: {len(our_features)}")
        print(f"  All available features: {len(all_feature_columns)}")
        
        # Show sample of features in each set
        print(f"\nSample of paper's features: {paper_16_features[:5]}...")
        print(f"Sample of our features: {our_features[:5]}...")

        return {
            'paper_16': paper_16_features,
            'our_enhanced': our_features,
            'all': all_feature_columns  # All features except target and metadata
        }

    def prepare_features(self, df, feature_set):
        """Prepare features for modeling with proper imputation and scaling"""
        
        # Select features
        X = df[feature_set].copy()
        
        # Handle missing values as per paper
        # For features where NA has meaning, we already created indicators
        # For others, use median imputation
        X_filled = self.imputer.fit_transform(X)
        
        # Scale features (important for SVM, LR as per paper)
        X_scaled = self.scaler.fit_transform(X_filled)
        
        return X_scaled, self.imputer, self.scaler
    
    def _emp_to_numeric(self, val):
        """Convert employment length to numeric (paper's method)"""
        if pd.isna(val):
            return np.nan
        
        val = str(val).lower()
        
        if '10+' in val:
            return 10
        elif '< 1' in val:
            return 0
        elif '1 year' in val:
            return 1
        else:
            # Extract numbers
            numbers = re.findall(r'\d+', val)
            if numbers:
                return float(numbers[0])
            else:
                return np.nan
    
    def calculate_feature_importance(self, model, feature_names):
        """Calculate and display feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Create feature importance dataframe
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feat_imp