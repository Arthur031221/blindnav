# 修復 NumPy 版本問題

## 問題說明
系統使用 NumPy 2.2.6，但 YOLO 和 EasyOCR 是用 NumPy 1.x 編譯的，會導致：
- "Numpy is not available" 錯誤
- 視訊流黑屏
- 系統不穩定

## 解決方法

### 方法 1：使用修復腳本（推薦）

**Windows:**
```bash
cd backend
.\fix_numpy.bat
```

**Linux/Mac/Git Bash:**
```bash
cd backend
bash fix_numpy.sh
```

### 方法 2：手動修復

1. **停止後端服務**（如果正在運行，按 Ctrl+C）

2. **降級 NumPy:**
   ```bash
   cd backend
   venv\Scripts\activate
   pip install "numpy<2.0"
   ```

3. **驗證版本:**
   ```bash
   python -c "import numpy; print('NumPy version:', numpy.__version__)"
   ```
   應該顯示類似：`NumPy version: 1.26.x`

4. **重新啟動後端服務:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## 修復後的效果
- ✅ 消除 NumPy 版本警告
- ✅ 解決 "Numpy is not available" 錯誤
- ✅ 修復視訊流黑屏問題
- ✅ 提升系統穩定性

