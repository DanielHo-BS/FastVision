# FastVision

## Overview
This project demonstrates how to accelerate image classification inference by converting a PyTorch model to ONNX and deploying it using FastAPI. A simple web UI allows users to upload images and compare inference results and speed between PyTorch and ONNX Runtime.

## Features
- Convert a **ResNet-18** model from PyTorch to ONNX
- Deploy a **FastAPI** server for image classification
- Web UI for uploading images and comparing inference results
- Performance comparison between PyTorch and ONNX Runtime
- Modular design with separate model initialization system

## Project Structure
- `main.py` - FastAPI server implementation
- `model.py` - Model loading and inference functionality
- `init_model.py` - Model initialization and environment setup
- `react-app/` - Frontend web interface

## Installation

### Prerequisites
- Python 3.12 (required for onnxruntime compatibility)
- UV (Python package installer)

### Install UV
First, install UV using pip:
```bash
pip install uv
```

### Install Dependencies
You can install dependencies using UV in two ways:

#### Method 1: Direct Installation
```bash
uv add fastapi uvicorn torch torchvision onnx onnxruntime pillow numpy
```

#### Method 2: Using [pyproject.toml](./pyproject.toml)
```bash
uv sync
```

Note: This project requires Python 3.12 as onnxruntime only supports Python 3.10 and above.

## Running the Project

### Using the Makefile
This project includes a Makefile to simplify common operations:

```bash
# Install all dependencies (Python and React)
make install

# Run only the backend FastAPI server
make run-backend

# Run only the frontend React server
make run-frontend

# Check available models and initialization
make init-model

# Run both frontend and backend servers simultaneously
make run

# Clean up generated files
make clean

# View all available commands
make help
```

### Manual Steps
If you prefer not to use the Makefile, you can follow these steps:

#### 1. Export PyTorch Model to ONNX
Run the following script to export a pre-trained ResNet-18 model to ONNX:
```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx")
```

#### 2. Start FastAPI Server
```bash
uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

#### 3. Access the Web UI
Open your browser and navigate to:
```
http://127.0.0.1:8000
```
Upload an image and view the classification results with inference times for PyTorch and ONNX.

## API Endpoints
### 1. Upload an Image and Get Predictions
**Endpoint:**
```
POST /predict/
```
**Request:**
- Upload an image file (`.jpg`, `.png`, etc.)

**Response:**
```json
{
    "filename": "image.jpg",
    "model": "resnet18",
    "pytorch_time": 0.12,
    "onnx_time": 0.08,
    "pytorch_duration": 0.12,
    "onnx_duration": 0.08,
    "onnx_speedup": 1.5,
    "pytorch_label": "dog",
    "onnx_label": "dog"
}
```

### 2. Get Available Models
**Endpoint:**
```
GET /available-models/
```

**Response:**
```json
{
    "models": ["resnet18", "resnet50", "vgg16", "vision_transformer"]
}
```

## Performance Comparison
The following table shows detailed performance benchmarks for different models:

### CPU Performance
| Model | PyTorch Inference Time (s) | ONNX Inference Time (s) | Speedup |
|--------|----------------------|----------------------|--------|
| ResNet-18 | 0.013804 | 0.005321 | 2.59x |
| ResNet-50 | 0.032168 | 0.012607 | 2.55x |
| VGG16 | 0.090777 | 0.048402 | 1.88x |
| Vision Transformer | 0.105784 | 0.063510 | 1.67x |

### GPU Performance
| Model | PyTorch Inference Time (s) | ONNX Inference Time (s) | Speedup |
|--------|----------------------|----------------------|--------|
| ResNet-18 | 0.005254 | 0.005265 | 1.00x |
| ResNet-50 | 0.004976 | 0.012051 | 0.41x |
| VGG16 | 0.005242 | 0.054336 | 0.10x |
| Vision Transformer | 0.006881 | 0.059524 | 0.12x |

### Key Findings:
- **CPU**: ONNX Runtime consistently delivers faster inference, with ResNet models showing the best optimization (over 2.5x speedup).
- **GPU**: PyTorch outperforms ONNX Runtime for most models except ResNet-18 (where they are equal).
- ONNX optimization benefits are primarily seen in CPU deployments.

All measurements were performed averaging 100 inference runs per model.

## Model Architecture
The project uses a modular design:
- `init_model.py` - Handles model initialization and provides the list of available models
- `model.py` - Manages model loading, ONNX conversion, and inference
- All models use pretrained ImageNet weights for classification

## Future Improvements
- Support for more image classification models
- Deploy as a Docker container
- Integrate with cloud-based inference services

## License
This project is licensed under the MIT License.

---
Developed for PyCon Taiwan Demo 🎉

