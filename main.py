from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
from model import Model
import argparse

# 初始化 FastAPI
app = FastAPI()

# 允許 CORS 跨域請求（讓 React 前端可以存取）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設定裝置
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', required=False)
args, _ = parser.parse_known_args()
model = Model(args)
model.convert_to_onnx()

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# API：處理影像上傳並執行 PyTorch 和 ONNX 推論
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(model.device)

    # PyTorch 推論
    start_time = time.time()
    pytorch_time = model.inference(img_tensor)
    pytorch_duration = time.time() - start_time
    pytorch_label = model.predict(img_tensor)

    # ONNX 推論
    start_time = time.time()
    onnx_time = model.inference_onnx(img_tensor)
    onnx_duration = time.time() - start_time
    onnx_label = model.predict_onnx(img_tensor)
    return JSONResponse({
        "filename": file.filename,
        "pytorch_time": round(pytorch_time, 4),
        "onnx_time": round(onnx_time, 4),
        "pytorch_duration": round(pytorch_duration, 4),
        "onnx_duration": round(onnx_duration, 4),
        "onnx_speedup": round(pytorch_time / onnx_time, 2),
        "pytorch_label": pytorch_label,
        "onnx_label": onnx_label
    })

# 啟動 FastAPI：
# uvicorn main:app --reload
