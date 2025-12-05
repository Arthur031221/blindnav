# GitHub 推送指南

## 步驟 1：設定 Git 用戶資訊

在終端機執行以下命令（替換成你的資訊）：

```bash
git config --global user.name "你的名字"
git config --global user.email "your.email@example.com"
```

或者只為這個專案設定（不影響其他專案）：

```bash
cd backend
git config user.name "你的名字"
git config user.email "your.email@example.com"
```

## 步驟 2：建立初始提交

```bash
cd /d/blindnav_local
git commit -m "Initial commit: BlindNav 視障者公車導航系統"
```

## 步驟 3：在 GitHub 上建立新倉庫

1. 前往 https://github.com/new
2. 輸入倉庫名稱（例如：`blindnav`）
3. 選擇公開或私有
4. **不要**勾選「Initialize this repository with a README」（因為我們已經有 README）
5. 點擊「Create repository」

## 步驟 4：連接到 GitHub 並推送

GitHub 會顯示類似這樣的指令，執行：

```bash
cd /d/blindnav_local
git remote add origin https://github.com/你的用戶名/你的倉庫名.git
git branch -M main
git push -u origin main
```

**注意**：如果使用 HTTPS，可能需要輸入 GitHub 用戶名和 Personal Access Token（不是密碼）。

### 使用 SSH（推薦）

如果你已經設定 SSH 金鑰：

```bash
git remote add origin git@github.com:你的用戶名/你的倉庫名.git
git branch -M main
git push -u origin main
```

## 步驟 5：驗證

前往你的 GitHub 倉庫頁面，應該能看到所有文件已經上傳。

## 後續更新

之後如果有修改，使用以下命令推送：

```bash
git add .
git commit -m "你的提交訊息"
git push
```

## 注意事項

- `.gitignore` 已經設定好，會自動排除：
  - `venv/`（虛擬環境）
  - `debug_bus_rois/`（除錯圖片）
  - `*.pt`（模型文件）
  - `__pycache__/`（Python 快取）
  - 其他不需要版本控制的文件

- 如果 `yolov8n.pt` 模型文件很大，建議不要上傳（已經在 .gitignore 中排除）



