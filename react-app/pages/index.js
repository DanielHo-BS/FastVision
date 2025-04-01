import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file)); // 產生圖片預覽
  };

  const handleUpload = async () => {
    if (!image) return alert("請選擇圖片！");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

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
      <input type="file" onChange={handleFileChange} />
      {preview && <img src={preview} alt="預覽圖片" className="preview" />}
      <button onClick={handleUpload} disabled={loading}>
        {loading ? "推論中..." : "上傳並分類"}
      </button>

      {result && (
        <div className="result">
          <h2>結果</h2>
          <p>檔案名稱: {result.filename}</p>
          <p>PyTorch 推論時間: {result.pytorch_duration} 秒</p>
          <p>ONNX 推論時間: {result.onnx_duration} 秒</p>
          <p>ONNX 加速倍數: {result.onnx_speedup} 倍</p>
        </div>
      )}

      <style jsx>{`
        .container {
          text-align: center;
          padding: 20px;
        }
        .preview {
          max-width: 300px;
          margin: 10px;
          border-radius: 8px;
        }
        .result {
          margin-top: 20px;
          padding: 10px;
          background: #f4f4f4;
          border-radius: 8px;
          display: inline-block;
          color: black;
        }
      `}</style>
    </div>
  );
}
