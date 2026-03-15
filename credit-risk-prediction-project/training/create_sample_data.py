"""
Create sample Lending Club data for testing TabTransformer
Run this if you can't download the real dataset from Kaggle
"""
import numpy as np
import pandas as pd
from pathlib import Path

def create_sample_data(n_samples=10000):
    """Create realistic sample lending data"""
    
    dest_dir = Path(__file__).parent / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = dest_dir / "lending_club_sample.csv"
    
    print("=" * 60)
    print("Creating Sample Lending Club Data")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate realistic distributions
    data = {
        # Loan characteristics
        'loan_amnt': np.random.lognormal(mean=9.5, sigma=0.5, size=n_samples).clip(1000, 40000),
        'int_rate': np.random.uniform(5, 30, n_samples),
        
        # Borrower characteristics  
        'annual_inc': np.random.lognormal(mean=11, sigma=0.6, size=n_samples).clip(20000, 500000),
        'dti': np.random.beta(2, 5, n_samples) * 50,  # Most DTI is low
        
        # Credit history
        'delinq_2yrs': np.random.choice([0, 0, 0, 0, 0, 1, 2], n_samples),
        'inq_last_6mths': np.random.choice([0, 0, 0, 1, 1, 2, 3], n_samples),
        'open_acc': np.random.randint(2, 30, n_samples),
        'pub_rec': np.random.choice([0, 0, 0, 0, 0, 0, 1], n_samples),
        'revol_bal': np.random.lognormal(mean=8, sigma=1, size=n_samples).clip(0, 100000),
        'total_acc': np.random.randint(3, 60, n_samples),
        'collections_12_mths_ex_med': np.random.choice([0, 0, 0, 0, 0, 1], n_samples),
        'acc_now_delinq': np.random.choice([0, 0, 0, 0, 0, 1], n_samples),
        'tot_coll_amt': np.random.choice([0, 0, 0, 0, 100, 500, 1000], n_samples),
        'tot_cur_bal': np.random.lognormal(mean=10, sigma=1, size=n_samples).clip(0, 500000),
        'total_rev_hi_lim': np.random.lognormal(mean=9, sigma=0.8, size=n_samples).clip(1000, 200000),
        
        # Categorical features
        'term': np.random.choice([' 36 months', ' 60 months'], n_samples, p=[0.75, 0.25]),
        
        # Target - realistic default rate ~15-20%
        'loan_status': np.random.choice(
            ['Fully Paid', 'Charged Off', 'Default'], 
            n_samples,
            p=[0.82, 0.15, 0.03]
        ),
        
        # Date - spread across 2013-2014
        'issue_d': np.random.choice([
            'Jan-2013', 'Feb-2013', 'Mar-2013', 'Apr-2013', 'May-2013', 'Jun-2013',
            'Jul-2013', 'Aug-2013', 'Sep-2013', 'Oct-2013', 'Nov-2013', 'Dec-2013',
            'Jan-2014', 'Feb-2014', 'Mar-2014', 'Apr-2014', 'May-2014', 'Jun-2014',
            'Jul-2014', 'Aug-2014', 'Sep-2014', 'Oct-2014', 'Nov-2014', 'Dec-2014'
        ], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Round numerical columns
    df['loan_amnt'] = df['loan_amnt'].round(0)
    df['int_rate'] = df['int_rate'].round(2)
    df['annual_inc'] = df['annual_inc'].round(0)
    df['dti'] = df['dti'].round(2)
    df['revol_bal'] = df['revol_bal'].round(0)
    df['tot_cur_bal'] = df['tot_cur_bal'].round(0)
    df['total_rev_hi_lim'] = df['total_rev_hi_lim'].round(0)
    
    # Save
    df.to_csv(sample_path, index=False)
    
    # Print summary
    size_mb = sample_path.stat().st_size / (1024 * 1024)
    default_rate = (df['loan_status'].isin(['Charged Off', 'Default'])).mean()
    
    print(f"\n✓ Created: {sample_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Default rate: {default_rate:.1%}")
    
    print(f"\nLoan status distribution:")
    print(df['loan_status'].value_counts())
    
    print(f"\n✓ Ready to train! Run:")
    print(f"  python train_model.py")
    
    return sample_path

if __name__ == "__main__":
    create_sample_data(n_samples=10000)
