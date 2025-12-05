# You are my eyes - 視障者公車導航系統

一個專為視障者設計的公車導航系統，結合 GPS 導航和 AI 公車號碼識別功能。

## 功能特色

- 🗺️ **Google Maps 路線規劃**：整合 Google Maps API，提供完整的公車路線規劃
- 📱 **手機攝像頭整合**：使用手機後置鏡頭進行即時影像識別
- 🤖 **AI 公車號碼識別**：使用 YOLO + EasyOCR 自動識別公車前方的 LED 顯示號碼
- 🔊 **語音導航**：完整的語音提示系統，引導視障者到達目的地
- 💻 **即時監視介面**：電腦端即時顯示辨識結果和影像流

## 技術架構

### 後端
- **FastAPI**：RESTful API 服務
- **YOLOv8**：物體偵測（識別公車）
- **EasyOCR**：文字識別（識別公車號碼）
- **OpenCV**：影像處理
- **PyTorch**：深度學習框架（支援 GPU 加速）

### 前端
- **HTML/CSS/JavaScript**：響應式網頁介面
- **WebRTC/MediaStream API**：手機攝像頭串流
- **Web Speech API**：語音合成

## 系統需求

- Python 3.12+
- NVIDIA GPU（可選，用於加速 AI 推理）
- CUDA 11.6+（如果使用 GPU）
- 現代瀏覽器（支援 WebRTC 和 MediaStream API）

## 安裝步驟

### 1. 克隆專案

```bash
git clone <your-repo-url>
cd blindnav_local
```

### 2. 設定虛擬環境

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

**注意**：如果需要 GPU 加速，請安裝 CUDA 版本的 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. 設定環境變數

創建 `.env` 文件（可選）：

```bash
GOOGLE_MAPS_API_KEY=your_api_key_here
```

### 5. 下載 YOLO 模型

YOLOv8n 模型會自動下載，或手動下載到 `backend/yolov8n.pt`

## 使用說明

### 啟動後端服務

```bash
cd backend
venv\Scripts\activate  # Windows
# 或
source venv/bin/activate  # Linux/Mac

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 使用 ngrok 建立 HTTPS 隧道（手機端訪問）

詳見 [start_with_ngrok.md](backend/start_with_ngrok.md)

### 開啟網頁介面

- **手機端**：`http://your-server-ip:8000` 或 ngrok HTTPS URL
- **電腦端監視**：`http://localhost:8000/bus_monitor`

## GPU 加速設定

如果你的系統有 NVIDIA GPU，可以啟用 GPU 加速以提升 AI 推理速度（5-10 倍提升）。

詳見 [start_with_ngrok.md](backend/start_with_ngrok.md) 中的「GPU 加速設定」章節。

## 專案結構

```
blindnav_local/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI 主應用
│   │   └── bus_vision.py    # 公車識別核心邏輯
│   ├── static/              # 前端靜態文件
│   │   ├── index.html       # 手機端主頁
│   │   ├── app.js           # 手機端邏輯
│   │   ├── monitor.html     # 電腦端監視頁面
│   │   ├── monitor.js       # 電腦端邏輯
│   │   ├── bus_monitor.html # 公車監視頁面
│   │   └── bus_monitor.js   # 公車監視邏輯
│   ├── debug_bus_rois/      # 偵測結果除錯圖片（不提交到 Git）
│   └── venv/                # 虛擬環境（不提交到 Git）
└── README.md
```

## 常見問題

### NumPy 版本問題

如果遇到 "Numpy is not available" 錯誤，請參考 [README_FIX_NUMPY.md](backend/README_FIX_NUMPY.md)

### 視訊流卡頓

- 確保已啟用 GPU 加速
- 檢查網路連線品質
- 降低 JPEG 品質設定（在 `bus_vision.py` 中）

## 授權

[你的授權資訊]

## 貢獻

歡迎提交 Issue 和 Pull Request！

