import { useState, useEffect } from "react";
import axios from "axios";

export default function Home() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");

  // 頁面載入時獲取可用模型列表
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/available-models/");
        setAvailableModels(response.data.models);
        // 設置默認選中的模型
        if (response.data.models.length > 0) {
          setSelectedModel(response.data.models[0]);
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };
    
    fetchModels();
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file)); // 產生圖片預覽
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleUpload = async () => {
    if (!image) return alert("請選擇圖片！");
    if (!selectedModel) return alert("請選擇模型！");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);
    formData.append("model_name", selectedModel);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict/", formData);
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      alert("推論失敗");
    }
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>影像分類 - PyTorch vs ONNX</h1>
      
      <div className="input-group">
        <label>選擇模型：</label>
        <select 
          value={selectedModel} 
          onChange={handleModelChange}
          disabled={loading}
        >
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>
      
      <div className="input-group">
        <label>選擇圖片：</label>
        <input type="file" onChange={handleFileChange} disabled={loading} />
      </div>
      
      {preview && <img src={preview} alt="預覽圖片" className="preview" />}
      
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "推論中..." : "上傳並分類"}
      </button>

      {result && (
        <div className="result">
          <h2>結果</h2>
          <p>檔案名稱: {result.filename}</p>
          <p>使用模型: {result.model}</p>
          <p>PyTorch 推論時間: {result.pytorch_duration} 秒</p>
          <p>ONNX 推論時間: {result.onnx_duration} 秒</p>
          <p>ONNX 加速倍數: {result.onnx_speedup} 倍</p>
          <p>PyTorch 預測: {result.pytorch_label}</p>
          <p>ONNX 預測: {result.onnx_label}</p>
        </div>
      )}

      <style jsx>{`
        .container {
          text-align: center;
          padding: 20px;
          max-width: 800px;
          margin: 0 auto;
        }
        .input-group {
          margin: 15px 0;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .input-group label {
          margin-right: 10px;
          font-weight: bold;
        }
        select {
          padding: 8px;
          border-radius: 4px;
          border: 1px solid #ccc;
          cursor: pointer;
        }
        .preview {
          max-width: 300px;
          margin: 10px;
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        button {
          background: #4a90e2;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 4px;
          cursor: pointer;
          font-size: 16px;
          margin-top: 10px;
        }
        button:hover {
          background: #357ae8;
        }
        button:disabled {
          background: #cccccc;
          cursor: not-allowed;
        }
        .result {
          margin-top: 20px;
          padding: 20px;
          background: #f4f4f4;
          border-radius: 8px;
          display: inline-block;
          color: black;
          text-align: left;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
      `}</style>
    </div>
  );
}
