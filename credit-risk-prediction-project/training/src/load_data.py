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

        # project root = credit-risk-prediction-project/training/src/.. /..
        self.project_root = Path(__file__).resolve().parents[2]
        self.training_dir = self.project_root / "training"

        raw = config["paths"]["raw_data"]

        if isinstance(raw, str) and raw.startswith("kaggle://"):
            self.kaggle_uri = raw
            filename = raw.split("/")[-1]
            self.data_path = self.training_dir / "data" / filename
        else:
            self.kaggle_uri = None
            raw_path = Path(raw)
            if raw_path.is_absolute():
                self.data_path = raw_path
            else:
                self.data_path = self.project_root / raw_path

        # Eagerly resolve: if the expected path is actually a directory
        # (Kaggle unzip can create accepted_2007_to_2018Q4.csv/ as a folder),
        # find the real CSV inside it now so we never re-trigger a download.
        if self.data_path.exists() and self.data_path.is_dir():
            csvs = sorted(
                self.data_path.rglob("*.csv"),
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if csvs:
                self.data_path = csvs[0]
                print(f"Resolved data path to: {self.data_path}")
    
    def _ensure_raw_data(self):
        import shutil
        import subprocess

        # Case 1: path already exists
        if self.data_path.exists():
            if self.data_path.is_file():
                return

            # If it exists but is a directory, try to resolve the real CSV inside it
            if self.data_path.is_dir():
                csvs = list(self.data_path.glob("*.csv")) + list(self.data_path.rglob("*.csv"))
                if csvs:
                    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
                    self.data_path = csvs[0]
                    print(f"Resolved directory to CSV: {self.data_path}")
                    return

                raise IsADirectoryError(
                    f"Expected a CSV file, but found a directory with no CSV inside: {self.data_path}"
                )

        # Case 2: file missing and no Kaggle source
        if not self.kaggle_uri:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Parse kaggle:///owner/dataset/file
        parts = self.kaggle_uri.replace("kaggle://", "").split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid Kaggle URI: {self.kaggle_uri}")

        dataset = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        dest_dir = self.data_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading from Kaggle: {dataset} (file={filename}) ...")

        kaggle_exe = shutil.which("kaggle")
        if not kaggle_exe:
            raise FileNotFoundError(
                "Kaggle CLI not found on PATH. Try: pip install kaggle and restart terminal."
            )

        subprocess.run(
            [kaggle_exe, "datasets", "download", "-d", dataset, "-p", str(dest_dir), "--unzip"],
            check=True
        )

        # After download, resolve the actual CSV
        if self.data_path.exists():
            if self.data_path.is_file():
                return
            if self.data_path.is_dir():
                csvs = list(self.data_path.glob("*.csv")) + list(self.data_path.rglob("*.csv"))
                if csvs:
                    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
                    self.data_path = csvs[0]
                    print(f"Resolved downloaded directory to CSV: {self.data_path}")
                    return

        csvs = list(dest_dir.glob("*.csv"))
        if not csvs:
            csvs = list(dest_dir.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found after Kaggle download into {dest_dir}")

        csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
        self.data_path = csvs[0]
        print(f"Using downloaded CSV: {self.data_path}")
    
    def _merge_external_macro_features(self, df):
        """
        Merge monthly external macro data to each loan using issue month.
        Expected files:
            data/external_macro/FEDFUNDS.csv
            data/external_macro/UNRATE.csv

        Each file is expected to have:
            observation_date,<series_name>
        e.g.
            observation_date,FEDFUNDS
            observation_date,UNRATE
        """
        # Gracefully skip if config doesn't define external macro data path
        macro_raw = self.config.get('paths', {}).get('external_macro_data')
        if not macro_raw:
            print("⚠️ 'paths.external_macro_data' not in config — skipping macro features.")
            return df

        macro_dir = Path(macro_raw)
        if not macro_dir.is_absolute():
            macro_dir = self.project_root / macro_dir

        fed_path = macro_dir / "FEDFUNDS.csv"
        unrate_path = macro_dir / "UNRATE.csv"

        if not fed_path.exists() or not unrate_path.exists():
            print(
                f"⚠️ External macro files not found: {fed_path} and/or {unrate_path}. "
                "Skipping external features."
            )
            return df

        # Read separate FRED files
        fed = pd.read_csv(fed_path)
        unrate = pd.read_csv(unrate_path)

        # Standardize column names
        fed = fed.rename(columns={"observation_date": "date", "FEDFUNDS": "fed_funds_rate"})
        unrate = unrate.rename(columns={"observation_date": "date", "UNRATE": "unemployment_rate"})

        # Parse dates and numeric values
        fed["date"] = pd.to_datetime(fed["date"], errors="coerce")
        unrate["date"] = pd.to_datetime(unrate["date"], errors="coerce")

        fed["fed_funds_rate"] = pd.to_numeric(fed["fed_funds_rate"], errors="coerce")
        unrate["unemployment_rate"] = pd.to_numeric(unrate["unemployment_rate"], errors="coerce")

        # Merge the two macro series together
        macro = fed.merge(unrate, on="date", how="outer").sort_values("date").reset_index(drop=True)

        # Forward fill missing monthly values if needed
        macro[["fed_funds_rate", "unemployment_rate"]] = (
            macro[["fed_funds_rate", "unemployment_rate"]].ffill()
        )

        # Engineer macro features on the macro table first
        macro["fed_funds_rate_3m_change"] = (
            macro["fed_funds_rate"] - macro["fed_funds_rate"].shift(3)
        )
        macro["unemployment_rate_3m_change"] = (
            macro["unemployment_rate"] - macro["unemployment_rate"].shift(3)
        )

        macro["rate_tightening_flag"] = (
            macro["fed_funds_rate_3m_change"] > 0
        ).astype("int8")

        macro["unemployment_rising_flag"] = (
            macro["unemployment_rate_3m_change"] > 0
        ).astype("int8")

        # Align loan issue dates to month start
        df = df.copy()
        df["issue_month"] = df["issue_date"].dt.to_period("M").dt.to_timestamp()

        # Merge macro features into loan data
        df = df.merge(
            macro,
            left_on="issue_month",
            right_on="date",
            how="left"
        )

        df = df.drop(columns=["date"], errors="ignore")

        print("✓ External macro features merged from FEDFUNDS.csv and UNRATE.csv")
        return df
        
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
        '''dtype_optimization = {
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
        }'''
        dtype_optimization = {
            'loan_amnt': 'float32',
            'int_rate': 'float32',
            'annual_inc': 'float32',
            'dti': 'float32',
            'revol_bal': 'float32',
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
        
        # Insert external macro features
        df = self._merge_external_macro_features(df)
        
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

        if 'revol_util' in df.columns:
            df['revol_util'] = (
                df['revol_util']
                .astype(str)
                .str.replace('%', '', regex=False)
                .replace('nan', np.nan)
            )
            df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
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