# TabTransformer Project - Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- git (optional, for version control)

## Installation Steps

### Option 1: Automated Setup (Recommended)

```bash
cd /Users/grace/fintech_new/CreditShield
chmod +x setup_env.sh
./setup_env.sh
```

### Option 2: Using Make

```bash
cd /Users/grace/fintech_new/CreditShield
make install
```

### Option 3: Manual Setup

```bash
# Navigate to project directory
cd /Users/grace/fintech_new/CreditShield

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Verification

After installation, verify everything works:

```bash
# Activate environment
source venv/bin/activate

# Check Python version
python --version

# Check key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import einops; print(f'Einops: {einops.__version__}')"
```

## Running Tests

```bash
# Activate environment
source venv/bin/activate

# Navigate to training directory
cd credit-risk-prediction-project/training

# Run all tests
bash run_tests.sh

# Or run individual tests
python src/test_tab_transformer.py      # Unit tests
python test_integration.py               # Integration tests
```

## Troubleshooting

### PyTorch Installation Issues

If you have an M1/M2 Mac, use:
```bash
pip install torch torchvision torchaudio
```

For GPU support on Linux/Windows:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

If you get memory errors:
1. Reduce batch_size in config
2. Reduce embedding_dim
3. Reduce model depth

### Missing Dependencies

If you see import errors:
```bash
pip install -r requirements.txt --force-reinstall
```

### Kaggle API Issues

To use Kaggle datasets:
1. Get API credentials from https://www.kaggle.com/settings/account
2. Create `~/.kaggle/kaggle.json` with your credentials
3. Or set environment variables:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

## Deactivating Environment

When done working:
```bash
deactivate
```

## Removing Environment

To clean up:
```bash
make clean
# or manually:
rm -rf venv
```

## Project Structure

```
CreditShield/
├── venv/                           # Virtual environment (auto-created)
├── requirements.txt                # Dependencies
├── setup_env.sh                    # Setup script
├── Makefile                        # Make commands
├── SETUP_GUIDE.md                  # This file
├── config/
│   └── tab_transformer_config.yaml # Model configuration
└── credit-risk-prediction-project/
    └── training/
        ├── src/
        │   ├── tab_transformer.py
        │   ├── train_tab_transformer.py
        │   ├── test_tab_transformer.py
        │   └── load_data.py
        ├── test_integration.py
        └── run_tests.sh
```
