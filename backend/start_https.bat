@echo off
REM 使用自簽名證書啟動 HTTPS 服務器
REM 需要先執行 generate_self_signed_cert.py 生成證書

cd /d %~dp0
call venv\Scripts\activate.bat

if not exist cert.pem (
    echo 證書不存在，正在生成...
    python generate_self_signed_cert.py
)

if not exist cert.pem (
    echo 證書生成失敗，請手動執行 generate_self_signed_cert.py
    pause
    exit /b 1
)

echo 啟動 HTTPS 服務器...
echo 注意：iOS 需要在設定中信任這個證書
uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem

pause


