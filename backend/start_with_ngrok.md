# 使用 ngrok 建立 HTTPS 隧道

## 步驟 1：安裝 ngrok
1. 前往 https://ngrok.com/download
2. 下載 Windows 版本
3. 解壓縮到任意目錄（例如：C:\ngrok）
4. 註冊 ngrok 帳號（免費）並取得 authtoken

## 步驟 2：設定 ngrok
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

## 步驟 3：啟動後端服務
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 步驟 4：在另一個終端啟動 ngrok
```bash
ngrok http 8000
```

## 步驟 5：使用 ngrok 提供的 HTTPS URL
ngrok 會顯示類似這樣的 URL：
```
Forwarding  https://xxxx-xxxx-xxxx.ngrok-free.app -> http://localhost:8000
```

在手機瀏覽器中使用這個 HTTPS URL 即可！

## GPU 加速設定（可選，但強烈建議）

### 檢查 GPU 狀態
✅ **GPU 已成功啟用！**
- GPU 型號：NVIDIA GeForce GTX 1050（3GB 記憶體）
- CUDA 版本：11.8（PyTorch 2.7.1+cu118）
- 狀態：CUDA available: True

**GPU 評價**：GTX 1050 雖然是較舊的 GPU（2016 年），但對於你的公車號碼識別任務來說完全足夠！可以帶來 5-10 倍的性能提升。

### 啟用 GPU 加速的步驟

#### 步驟 1：卸載現有的 CPU 版本 PyTorch
```bash
cd backend
venv\Scripts\activate
pip uninstall torch torchvision torchaudio -y
```

#### 步驟 2：安裝 CUDA 版本的 PyTorch
根據你的 CUDA 11.6，建議安裝 CUDA 11.8 版本的 PyTorch（相容性較好）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**注意**：如果上述指令失敗，可以嘗試 CUDA 12.1 版本（你的驅動程式可能也支援）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 步驟 3：驗證 GPU 是否可用
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果顯示 `CUDA available: True`，表示 GPU 已成功啟用！

#### 步驟 4：重新啟動服務
重新啟動後端服務後，YOLO 和 EasyOCR 會自動使用 GPU，影像識別速度會顯著提升（通常快 5-10 倍）。

### 性能提升預期
- **YOLO 物體偵測**：從 ~100-200ms/幀 降至 ~20-50ms/幀
- **EasyOCR 文字識別**：從 ~500-1000ms 降至 ~100-200ms
- **整體處理速度**：提升約 **5-10 倍**

### 注意事項
- GPU 記憶體使用：你的 GPU 有 3GB 記憶體，對於 YOLOv8n 和 EasyOCR 來說應該足夠
- 如果遇到記憶體不足，可以降低 `_DETECT_INTERVAL` 的值（在 `bus_vision.py` 中）
- 使用 GPU 時會增加功耗和發熱，這是正常現象


