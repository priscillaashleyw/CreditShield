"""
load_data.py - Data loading and preprocessing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import subprocess

class DataLoader:
    """Handles data loading and preprocessing with paper's methodology"""
    
    def __init__(self, config):
        self.config = config
        raw = config['paths']['raw_data']
        # If raw is a Kaggle URI, store it + set local cache path
        if isinstance(raw, str) and raw.startswith("kaggle://"):
            self.kaggle_uri = raw 
            filename = raw.split("/")[-1]
            self.data_path = Path("training/data") / filename
        else:
            self.kaggle_uri = None
            self.data_path = Path(raw)
    
    def _ensure_raw_data(self):
        if self.data_path.exists():
            return

        if not self.kaggle_uri:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Parse kaggle:/<owner>/<dataset>/<file>
        parts = self.kaggle_uri.replace("kaggle://", "").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid Kaggle URI: {self.kaggle_uri}")

        dataset = "/".join(parts[:2])   # owner/dataset
        filename = "/".join(parts[2:])  # handle any extra slashes safely
        dest_dir = self.data_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading from Kaggle: {dataset} (file={filename}) ...")
        import shutil

        kaggle_exe = shutil.which("kaggle")
        if not kaggle_exe:
            raise FileNotFoundError(
                "Kaggle CLI not found on PATH. Try: pip install kaggle and restart terminal."
            )

        try:
            result = subprocess.run(
                [kaggle_exe, "datasets", "download", "-d", dataset, "-p", str(dest_dir), "--unzip"],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Kaggle download failed: {e.stderr}")
            print("\n" + "=" * 60)
            print("MANUAL DOWNLOAD INSTRUCTIONS:")
            print("=" * 60)
            print(f"1. Go to: https://www.kaggle.com/datasets/{dataset}")
            print(f"2. Click 'Download' button")
            print(f"3. Extract the CSV file to: {dest_dir}")
            print(f"4. Rename the file to: {self.data_path.name}")
            print("=" * 60)
            print("\nAlternatively, set up Kaggle API credentials:")
            print("  1. Go to https://www.kaggle.com/settings/account")
            print("  2. Click 'Create New Token' under API section")
            print("  3. Save kaggle.json to ~/.kaggle/kaggle.json")
            print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
            print("=" * 60 + "\n")
            raise FileNotFoundError(
                f"Could not download dataset. Please download manually from "
                f"https://www.kaggle.com/datasets/{dataset}"
            )

        # After unzip, ensure the expected file exists
        if not self.data_path.exists():
            # fallback: pick any CSV in dest_dir
            csvs = list(dest_dir.glob("*.csv"))
            if not csvs:
                csvs = list(dest_dir.rglob("*.csv"))
            if not csvs:
                raise FileNotFoundError(f"No CSV found after Kaggle download into {dest_dir}")
            csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
            # Use the largest CSV (most likely the main data file)
            largest_csv = csvs[0]
            print(f"Found CSV: {largest_csv.name}, using as data source")
            if largest_csv != self.data_path:
                largest_csv.rename(self.data_path)
        
    def load_and_filter_data(self):
        """
        Load and filter data according to paper specifications
        """
        print("Loading data according to paper methodology...")
        
        # Check if file exists
        self._ensure_raw_data()
        
        # Load data with optimized memory usage
        essential_cols = self.config['data_settings']['essential_columns']
        years = self.config['data_settings']['years']
        
        # Define optimized dtypes for common columns
        dtype_optimization = {
            'loan_amnt': 'float32',
            'int_rate': 'float32',
            'annual_inc': 'float32',
            'dti': 'float32',
            'delinq_2yrs': 'int8',
            'inq_last_6mths': 'int8',
            'open_acc': 'int16',
            'pub_rec': 'int8',
            'revol_bal': 'float32',
            'total_acc': 'int16',
            'collections_12_mths_ex_med': 'int8',
            'acc_now_delinq': 'int8',
            'tot_coll_amt': 'float32',
            'tot_cur_bal': 'float32',
            'total_rev_hi_lim': 'float32'
        }
        
        try:
            print(f"Loading data with memory optimization...")
            # First, read just the column names to see what we have
            with open(self.data_path, 'r') as f:
                header = f.readline().strip().split(',')
            
            # Only load columns we need
            cols_to_load = [col for col in essential_cols if col in header]
            print(f"Loading {len(cols_to_load)} of {len(essential_cols)} essential columns")
            
            # Create dtype dict for columns we're loading
            dtypes = {}
            for col in cols_to_load:
                if col in dtype_optimization:
                    dtypes[col] = dtype_optimization[col]
            
            # Load data in chunks
            chunk_size = 100000
            chunks = []
            
            for chunk in pd.read_csv(
                self.data_path,
                usecols=cols_to_load,
                dtype=dtypes,
                low_memory=False,
                chunksize=chunk_size
            ):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"Error with optimized loading: {e}")
            print("Falling back to standard loading...")
            # Fallback to your original loading code
            df = pd.read_csv(
                self.data_path,
                usecols=lambda x: x in essential_cols,
                low_memory=False
            )
            print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Convert dates
        df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
        df['issue_year'] = df['issue_date'].dt.year
        
        # Filter by years 2013-2014 as per paper
        if years:
            mask_year = df['issue_year'].between(years[0], years[1])
            df = df[mask_year].copy()
            print(f"✓ Filtered to years {years}: {len(df):,} rows")
        
        # Remove loans with incomplete status (as per paper)
        incomplete_statuses = self.config['data_settings']['incomplete_statuses']
        mask_incomplete = ~df['loan_status'].isin(incomplete_statuses)
        df = df[mask_incomplete].copy()
        
        print(f"✓ Removed incomplete loans: {len(df):,} remaining")
        print(f"Date range: {df['issue_date'].min().date()} to {df['issue_date'].max().date()}")

        # Remove Post-Origination Leakage Columns (from config)
        leakage_cols = set(self.config['data_settings'].get('leaked_columns', []))

        present_leakage = leakage_cols.intersection(df.columns)
        if present_leakage:
            print(f"⚠️ Dropping leakage columns from config: {present_leakage}")
            df = df.drop(columns=list(present_leakage), errors="ignore")

        # Handle structural missingness for columns with high % of missing values ("months since last X" bureau events)
        # In this case, Missing usually means "never happened"
        structural_missing_cols = [
            "mths_since_last_delinq",
            "mths_since_last_major_derog",
            "mths_since_last_record",
        ]

        for col in structural_missing_cols:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype("int8")
                df[col] = df[col].fillna(999).astype("int16", errors="ignore")
        
        # Transform heavy-tailed variables
        if 'annual_inc' in df.columns:
            df['annual_inc'] = np.log1p(df['annual_inc'])

        if 'revol_bal' in df.columns:
            df['revol_bal'] = np.log1p(df['revol_bal'])

        # Clip utilization (robustness; outlier handling)
        if 'revol_util' in df.columns:
            df['revol_util'] = df['revol_util'].clip(upper=120)

        return df
    
    def define_target(self, df, strategy='business'):
        """Define target variable based on loan status"""
        
        # Get target settings from config
        target_settings = self.config['target_settings']
        
        # Validate strategy
        if strategy not in target_settings:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(target_settings.keys())}")
        
        # Get the mapping for the chosen strategy
        mapping = target_settings[strategy]
        default_statuses = mapping['default_statuses']
        good_statuses = mapping['good_statuses']
        
        print(f"Defining target using '{strategy}' strategy...")
        print(f"  Default statuses: {default_statuses}")
        print(f"  Good statuses: {good_statuses}")
        
        # Filter loans with known outcomes
        mask = df['loan_status'].isin(good_statuses + default_statuses)
        df_clean = df[mask].copy()
        
        # Create target: 1 = default, 0 = good
        df_clean['target'] = df_clean['loan_status'].apply(
            lambda x: 1 if x in default_statuses else 0
        )
        
        # Calculate statistics
        n_loans = len(df_clean)
        n_defaults = df_clean['target'].sum()
        default_rate = n_defaults / n_loans if n_loans > 0 else 0
        
        print(f"✓ Target defined:")
        print(f"  Total loans: {n_loans:,}")
        print(f"  Defaults: {n_defaults:,} ({default_rate:.2%})")
        print(f"  Good/Default ratio: {(n_loans - n_defaults)/max(n_defaults, 1):.1f}:1")
        print(f"  Removed {len(df) - len(df_clean):,} loans with unknown status")
        
        return df_clean
    
    def time_based_split(self, df, train_ratio=0.7, val_ratio=0.15):
        """
        Split data chronologically for realistic validation
        (paper uses random split, but we improve with time-based)
        """
        if 'issue_date' not in df.columns:
            df['issue_date'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
        
        # Sort by date
        df_sorted = df.sort_values('issue_date').reset_index(drop=True)
        
        # Calculate split indices
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split chronologically
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        print(f"\n✓ Time-based split completed:")
        print(f"  Training: {len(train_df):,} loans ({train_df['issue_date'].min().date()} to {train_df['issue_date'].max().date()})")
        print(f"  Validation: {len(val_df):,} loans ({val_df['issue_date'].min().date()} to {val_df['issue_date'].max().date()})")
        print(f"  Testing: {len(test_df):,} loans ({test_df['issue_date'].min().date()} to {test_df['issue_date'].max().date()})")
        
        return train_df, val_df, test_df
    
    def random_split(self, df, test_size=0.2, random_state=42):
        """Random split (paper's method) - returns X_train, X_test, y_train, y_test"""
        from sklearn.model_selection import train_test_split
        
        # Prepare features and target
        if 'target' not in df.columns:
            raise ValueError("DataFrame must have 'target' column. Call define_target() first.")
        
        y = df['target']
        X = df.drop(['target', 'loan_status'], axis=1, errors='ignore')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Random split completed:")
        print(f"  Training: {len(X_train):,} samples")
        print(f"  Testing: {len(X_test):,} samples")
        print(f"  Training default rate: {y_train.mean():.2%}")
        print(f"  Testing default rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test