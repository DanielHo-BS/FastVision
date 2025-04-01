# Makefile for FastVision project

# Default target
all: clean

# Clean all ONNX files
clean-onnx:
	@echo "Cleaning ONNX files..."
	rm -f *.onnx
	@echo "ONNX files cleaned successfully"

# Clean Python cache files
clean-python:
	@echo "Cleaning Python cache files..."
	rm -rf __pycache__/
	rm -f *.pyc
	rm -rf .pytest_cache/
	@echo "Python cache files cleaned successfully"

# Clean virtual environment
clean-venv:
	@echo "Cleaning virtual environment..."
	rm -rf .venv/
	@echo "Virtual environment cleaned successfully"

# Clean all artifacts (ONNX, Python cache, and virtual environment)
clean: clean-onnx clean-python clean-venv
	@echo "All artifacts cleaned successfully"

# Install dependencies
install:
	@echo "Installing dependencies..."
	python -m pip install -r requirements.txt
	@echo "Dependencies installed successfully"

# Run the FastAPI server
run:
	@echo "Starting FastAPI server..."
	uvicorn main:app --reload

# Run tests
test:
	@echo "Running tests..."
	python -m pytest

# Show help message
help:
	@echo "Available targets:"
	@echo "  clean-onnx  - Remove all ONNX files"
	@echo "  clean-python - Remove Python cache files"
	@echo "  clean-venv  - Remove virtual environment"
	@echo "  clean      - Remove all artifacts (ONNX, Python cache, and venv)"
	@echo "  install    - Install project dependencies"
	@echo "  run        - Start the FastAPI server"
	@echo "  test       - Run tests"
	@echo "  help       - Show this help message"

.PHONY: all clean clean-onnx clean-python clean-venv install run test help 