# Makefile for cleaning ONNX files and related artifacts

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
	@echo "Python cache files cleaned successfully"

# Clean all artifacts (ONNX files and Python cache)
clean: clean-onnx clean-python
	@echo "All artifacts cleaned successfully"

# Show help message
help:
	@echo "Available targets:"
	@echo "  clean-onnx  - Remove all ONNX files"
	@echo "  clean-python - Remove Python cache files"
	@echo "  clean      - Remove all artifacts (ONNX and Python cache)"
	@echo "  help       - Show this help message"

.PHONY: all clean clean-onnx clean-python help 