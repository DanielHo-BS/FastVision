# FastVision

## Overview
This project demonstrates how to accelerate image classification inference by converting a PyTorch model to ONNX and deploying it using FastAPI. A simple web UI allows users to upload images and compare inference results and speed between PyTorch and ONNX Runtime.

## Features
- Convert a **ResNet-18** model from PyTorch to ONNX
- Deploy a **FastAPI** server for image classification
- Web UI for uploading images and comparing inference results
- Performance comparison between PyTorch and ONNX Runtime

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
### 1. Export PyTorch Model to ONNX
Run the following script to export a pre-trained ResNet-18 model to ONNX:
```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx")
```

### 2. Start FastAPI Server
```bash
uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

### 3. Access the Web UI
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
    "pytorch_prediction": "dog",
    "onnx_prediction": "dog",
    "pytorch_time": 0.12,
    "onnx_time": 0.08
}
```

## Performance Comparison
| Model | Inference Time (seconds) |
|--------|----------------------|
| PyTorch | 0.12 |
| ONNX | 0.08 |

ONNX is faster due to optimized inference execution!

## Future Improvements
- Support for more image classification models
- Deploy as a Docker container
- Integrate with cloud-based inference services

## License
This project is licensed under the MIT License.

---
Developed for PyCon Taiwan Demo 🎉

