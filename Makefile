.PHONY: help env activate install test clean

help:
	@echo "Available commands:"
	@echo "  make env       - Create virtual environment"
	@echo "  make activate  - Activate virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make test      - Run all tests"
	@echo "  make clean     - Remove virtual environment"

env:
	python3 -m venv venv
	@echo "✓ Virtual environment created"

activate:
	@echo "Run this command to activate:"
	@echo "  source venv/bin/activate"

install: env
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✓ Dependencies installed"

test:
	. venv/bin/activate && cd credit-risk-prediction-project/training && bash run_tests.sh

clean:
	rm -rf venv
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	@echo "✓ Virtual environment and cache removed"
