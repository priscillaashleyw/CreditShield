#!/bin/bash

set -e  # Exit on any error

echo "=============================================="
echo "Setting up Virtual Environment"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${BLUE}✓ Found Python ${PYTHON_VERSION}${NC}"

# Create virtual environment
if [ -d "venv" ]; then
    echo -e "${BLUE}Virtual environment already exists. Removing...${NC}"
    rm -rf venv
fi

echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

echo ""
echo "=============================================="
echo -e "${GREEN}✓ Virtual environment setup complete!${NC}"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo -e "${BLUE}  source venv/bin/activate${NC}"
echo ""
echo "To deactivate, run:"
echo -e "${BLUE}  deactivate${NC}"
echo ""
echo "To run tests:"
echo -e "${BLUE}  cd credit-risk-prediction-project/training${NC}"
echo -e "${BLUE}  bash run_tests.sh${NC}"
echo ""
