"""
Download Lending Club data from Kaggle
Run this script to download data before training.
"""
import subprocess
import sys
from pathlib import Path

def download_lending_club_data():
    """Download Lending Club dataset from Kaggle"""
    
    # Dataset options (try in order)
    datasets = [
        ("wordsforthewise/lending-club", "Lending Club (wordsforthewise)"),
        ("huseyinelci/lending-club-loan-data-20072020", "Lending Club 2007-2020"),
    ]
    
    dest_dir = Path(__file__).parent / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Lending Club Data Downloader")
    print("=" * 60)
    print(f"Target directory: {dest_dir.absolute()}")
    
    # Check for existing data
    existing_csvs = list(dest_dir.glob("*.csv"))
    if existing_csvs:
        print(f"\n✓ Found existing data files:")
        for csv in existing_csvs:
            size_mb = csv.stat().st_size / (1024 * 1024)
            print(f"  - {csv.name} ({size_mb:.1f} MB)")
        print("\n✓ Data already exists. Ready to train!")
        print(f"  Run: python train_model.py")
        return True
    
    print("\nNo existing data found. Attempting to download from Kaggle...")
    print("\nPrerequisites:")
    print("  1. Kaggle account")
    print("  2. API token in ~/.kaggle/kaggle.json")
    print("  3. kaggle CLI installed (pip install kaggle)")
    
    # Check if kaggle CLI is available
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"\n✓ Kaggle CLI found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("\n✗ Kaggle CLI not found!")
        print("  Install with: pip install kaggle")
        print_manual_instructions(datasets, dest_dir)
        return False
    
    # Check for kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"\n✗ Kaggle credentials not found at {kaggle_json}")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New Token'")
        print("  3. Save to ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print_manual_instructions(datasets, dest_dir)
        return False
    else:
        print(f"✓ Kaggle credentials found at {kaggle_json}")
    
    # Try to download
    for dataset_id, dataset_name in datasets:
        print(f"\nTrying: {dataset_name}...")
        print(f"  Dataset: https://www.kaggle.com/datasets/{dataset_id}")
        
        try:
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(dest_dir), "--unzip"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(f"  stdout: {result.stdout[:200] if result.stdout else '(empty)'}")
            print(f"  stderr: {result.stderr[:200] if result.stderr else '(empty)'}")
            
            if result.returncode == 0:
                print(f"\n✓ Successfully downloaded: {dataset_name}")
                
                # List downloaded files
                csvs = list(dest_dir.glob("*.csv"))
                if csvs:
                    print(f"\nDownloaded files:")
                    for csv in csvs:
                        size_mb = csv.stat().st_size / (1024 * 1024)
                        print(f"  - {csv.name} ({size_mb:.1f} MB)")
                    print(f"\n✓ Ready to train! Run: python train_model.py")
                    return True
                else:
                    print("  Warning: No CSV files found after download")
            else:
                print(f"  Download failed (exit code {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print("  Timeout - download taking too long")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Manual download instructions
    print_manual_instructions(datasets, dest_dir)
    return False

def print_manual_instructions(datasets, dest_dir):
    """Print manual download instructions"""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("\nAutomatic download failed. Please download manually:")
    print("\n1. Go to one of these URLs:")
    for dataset_id, dataset_name in datasets:
        print(f"   https://www.kaggle.com/datasets/{dataset_id}")
    print(f"\n2. Click 'Download' button (you may need to sign in)")
    print(f"\n3. Extract the ZIP file to:")
    print(f"   {dest_dir.absolute()}")
    print("\n4. Run train_model.py again:")
    print("   python train_model.py")
    print("=" * 60)

def create_sample_data():
    """Create sample data for testing if real data unavailable"""
    import numpy as np
    import pandas as pd
    
    dest_dir = Path(__file__).parent / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = dest_dir / "sample_lending_club.csv"
    
    print("\n" + "=" * 60)
    print("Creating SAMPLE data for testing...")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'loan_amnt': np.random.uniform(1000, 40000, n_samples),
        'int_rate': np.random.uniform(5, 30, n_samples),
        'annual_inc': np.random.uniform(20000, 200000, n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'delinq_2yrs': np.random.choice([0, 0, 0, 1, 2], n_samples),
        'inq_last_6mths': np.random.choice([0, 0, 1, 1, 2, 3], n_samples),
        'open_acc': np.random.randint(1, 30, n_samples),
        'pub_rec': np.random.choice([0, 0, 0, 0, 1], n_samples),
        'revol_bal': np.random.uniform(0, 50000, n_samples),
        'total_acc': np.random.randint(1, 50, n_samples),
        'collections_12_mths_ex_med': np.random.choice([0, 0, 0, 0, 1], n_samples),
        'acc_now_delinq': np.random.choice([0, 0, 0, 0, 1], n_samples),
        'tot_coll_amt': np.random.choice([0, 0, 0, 100, 500], n_samples),
        'tot_cur_bal': np.random.uniform(0, 200000, n_samples),
        'total_rev_hi_lim': np.random.uniform(1000, 100000, n_samples),
        'term': np.random.choice([' 36 months', ' 60 months'], n_samples),
        'loan_status': np.random.choice(
            ['Fully Paid', 'Fully Paid', 'Fully Paid', 'Charged Off', 'Default'], 
            n_samples
        ),
        'issue_d': np.random.choice(['Jan-2013', 'Jun-2013', 'Jan-2014', 'Jun-2014'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(sample_path, index=False)
    
    size_mb = sample_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Created sample data: {sample_path.name} ({size_mb:.2f} MB)")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\n⚠️  NOTE: This is SAMPLE data for testing only!")
    print("    For real results, download the actual Lending Club dataset.")
    print(f"\n✓ Ready to train! Run: python train_model.py")
    
    return True

if __name__ == "__main__":
    print()
    success = download_lending_club_data()
    
    if not success:
        print("\n" + "-" * 60)
        response = input("Would you like to create SAMPLE data for testing? (y/n): ").strip().lower()
        if response == 'y':
            try:
                import pandas as pd
                import numpy as np
                create_sample_data()
                success = True
            except ImportError as e:
                print(f"Error: {e}")
                print("Make sure pandas and numpy are installed")
        else:
            print("\nTo create sample data later, run:")
            print("  python -c \"from download_data import create_sample_data; create_sample_data()\"")
    
    sys.exit(0 if success else 1)
