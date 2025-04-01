from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
from model import Model
import argparse

# 初始化 FastAPI 應用
app = FastAPI()

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

# 1000 類別標籤（這裡用假資料）
labels = ["class_" + str(i) for i in range(1000)] 

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(model.device)

    # PyTorch 推論
    start_time = time.time()
    pytorch_time = model.inference(img_tensor)
    pytorch_duration = time.time() - start_time

    # ONNX 推論
    start_time = time.time()
    onnx_time = model.inference_onnx(img_tensor)
    onnx_duration = time.time() - start_time

    return JSONResponse({
        "filename": file.filename,
        "pytorch_time": round(pytorch_time, 4),
        "onnx_time": round(onnx_time, 4),
        "pytorch_duration": round(pytorch_duration, 4),
        "onnx_duration": round(onnx_duration, 4),
        "onnx_speedup": round(pytorch_time / onnx_time, 2)
    })

# 啟動 API 伺服器指令
# uvicorn main:app --reload
