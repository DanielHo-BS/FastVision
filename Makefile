# Makefile for FastVision project

# Default target
all: install

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

# Clean all artifacts (excluding virtual environment and node_modules)
clean: clean-onnx clean-python
	@echo "All artifacts cleaned successfully"
	@echo "Note: Virtual environment (.venv) and React dependencies (node_modules) are not cleaned"
	@echo "To clean these, please remove them manually if needed"

# Install uv if not already installed
install-uv:
	@echo "Checking for uv installation..."
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv..." && \
		pip install uv; \
	else \
		echo "uv is already installed"; \
	fi

# Install Python dependencies
install-python: install-uv
	@echo "Installing Python dependencies..."
	uv sync
	@echo "Python dependencies installed successfully"

# Install React dependencies
install-react:
	@echo "Installing React dependencies..."
	cd react-app && npm install
	@echo "React dependencies installed successfully"

# Install all dependencies
install: install-python install-react
	@echo "All dependencies installed successfully"

# Run the FastAPI server
run-backend:
	@echo "Starting FastAPI server..."
	uv run uvicorn main:app --reload

# Run the React development server
run-frontend:
	@echo "Starting React development server..."
	cd react-app && npm run dev

# Initialize model
init-model:
	@echo "Initializing model..."
	uv run python -c "from init_model import get_available_models; print('Available models:', get_available_models())"
	@echo "Model initialization check completed successfully"

# Run both frontend and backend servers in parallel
run:
	@echo "Starting both servers in parallel..."
	@echo "Press Ctrl+C to stop all servers"
	@trap 'kill 0' SIGINT; \
	uv run uvicorn main:app --reload & \
	cd react-app && npm run dev & \
	wait

# Build React for production
build-react:
	@echo "Building React for production..."
	cd react-app && npm run build
	@echo "React build completed successfully"

# Run tests
test:
	@echo "Running tests..."
	uv run pytest

# Show help message
help:
	@echo "Available targets:"
	@echo "  clean-onnx    - Remove all ONNX files"
	@echo "  clean-python  - Remove Python cache files"
	@echo "  clean        - Remove all artifacts (excluding virtual environment and node_modules)"
	@echo "  install-uv   - Install uv package manager"
	@echo "  install-python - Install Python dependencies using uv and pyproject.toml"
	@echo "  install-react  - Install React dependencies"
	@echo "  install      - Install all dependencies"
	@echo "  init-model   - Check model initialization and available models"
	@echo "  run-backend  - Start the FastAPI server"
	@echo "  run-frontend - Start the React development server"
	@echo "  run         - Start both frontend and backend servers in parallel"
	@echo "  build-react - Build React for production"
	@echo "  test        - Run tests"
	@echo "  help        - Show this help message"

.PHONY: all clean clean-onnx clean-python install install-uv install-python install-react init-model run run-backend run-frontend build-react test help 