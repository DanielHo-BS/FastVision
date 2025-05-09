from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
from model import Model
from init_model import get_available_models
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
parser.add_argument('--model', type=str, default='resnet18', required=False)
args, _ = parser.parse_known_args()

# 預先載入支援的模型
models = {}
available_models = get_available_models()

# 默認先載入 resnet18
default_model = available_models[0]
args.model = default_model
models[default_model] = Model(args)
models[default_model].convert_to_onnx()

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# API：獲取可用模型列表
@app.get("/available-models/")
async def get_available_models_api():
    return {"models": available_models}

# API：處理影像上傳並執行 PyTorch 和 ONNX 推論
@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Form(default=default_model)):
    # 驗證模型名稱
    if model_name not in available_models:
        return JSONResponse(
            status_code=400,
            content={"message": f"Invalid model name. Available models: {available_models}"}
        )
    
    # 如果模型尚未載入，則載入它
    if model_name not in models:
        args.model = model_name
        models[model_name] = Model(args)
        models[model_name].convert_to_onnx()
    
    # 使用指定的模型
    model = models[model_name]
    
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
        "model": model_name,
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
