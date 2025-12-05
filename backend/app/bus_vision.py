# app/bus_vision.py

import os
import cv2
import time
import threading
import re
from typing import Any, Dict, Optional

import numpy as np
from ultralytics import YOLO
import easyocr

# ---------------- 設定區 ----------------

# YOLO 權重：如果你有自己的 bus 模型，可以改這裡，例如 "bus_best.pt"
YOLO_WEIGHTS = "yolov8n.pt"

# YOLO conf 門檻：先設低一點，確保抓得到東西
# 降低門檻以增加敏感度，更容易偵測到公車
YOLO_CONF = 0.05  # 降低到 0.05，大幅增加敏感度，確保能偵測到物體

# 幀率控制：每幾張 frame 做一次 YOLO+OCR
_DETECT_INTERVAL = 3  # 每 3 張做一次偵測，減少運算量以提升影片流暢度

# 視訊流 FPS 限制（避免過度處理導致卡頓）
# 影片模式下使用較慢的 FPS，讓 YOLO 有足夠時間處理
_TARGET_FPS = 30  # 恢復到 30 FPS，讓影片看起來流暢
_FRAME_INTERVAL = 1.0 / _TARGET_FPS  # 每幀間隔時間（秒）

# 影片播放速度控制（固定為正常速度，不受模擬速度影響）
_video_fps: Optional[float] = None  # 影片的實際 FPS
_video_frame_interval: Optional[float] = None  # 根據影片 FPS 計算的 frame interval
_last_video_frame: Optional[np.ndarray] = None  # 緩存最後一幀，避免循環播放時閃爍
_video_start_sec: float = 0.0  # 影片開始時間（秒）
_video_end_sec: Optional[float] = None  # 影片結束時間（秒）

# Debug：是否把 bus LED ROI 存成圖片
DEBUG_SAVE_ROI = True
_DEBUG_SAVE_LIMIT = 200

# ---------------- 輸入來源設定 ----------------
# 設定輸入來源模式：
#   "realtime" - 使用即時影像（手機攝影機）
#   "video"    - 使用影片檔案
INPUT_MODE = "video"  # 改這裡選擇模式："realtime" 或 "video"

# 影片檔案路徑（僅在 INPUT_MODE = "video" 時使用）
# 範例路徑：
#   Windows: "D:/videos/bus_video.mp4" 或 r"D:\videos\bus_video.mp4"
#   Linux/Mac: "/home/user/videos/bus_video.mp4"
VIDEO_FILE_PATH = r"D:\blindnav_local\backend\videos\IMG_3361.MOV"  # 改這裡設定影片路徑

# ---------------- 模型初始化 ----------------

yolo_model = None
_yolo_error_count = 0
_yolo_last_error_time = 0

def _init_yolo_model():
    """初始化或重新初始化 YOLO 模型"""
    global yolo_model
    try:
        print(f"[bus_vision] Loading YOLO model from {YOLO_WEIGHTS} ...")
        yolo_model = YOLO(YOLO_WEIGHTS)
        
        # 嘗試使用 GPU（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[bus_vision] YOLO model will use GPU (CUDA device: {torch.cuda.get_device_name(0)})")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                print("[bus_vision] YOLO model will use GPU (Apple Silicon MPS)")
            else:
                device = "cpu"
                print("[bus_vision] YOLO model will use CPU (GPU not available)")
            
            # 將模型移到指定設備（YOLO 會自動處理，但我們可以明確指定）
            # YOLO 在推理時會自動使用可用的設備
        except ImportError:
            print("[bus_vision] PyTorch not available, using default device")
            device = "cpu"
        
        print("[bus_vision] YOLO model loaded.")
        return True
    except Exception as e:
        print(f"[bus_vision] Failed to load YOLO model: {e}")
        yolo_model = None
        return False

# 初始化模型
_init_yolo_model()

print("[bus_vision] Initializing EasyOCR (en only)...")
# 嘗試使用 GPU（如果可用），但考慮到 GPU 記憶體限制（3GB），優先使用 CPU
# 如果 GPU 記憶體不足，EasyOCR 會自動切換到 CPU
try:
    import torch
    if torch.cuda.is_available():
        # 檢查 GPU 記憶體大小
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[bus_vision] GPU 記憶體總量: {gpu_memory_gb:.2f} GB")
        
        # 如果 GPU 記憶體小於 4GB，優先使用 CPU 以避免 OOM
        if gpu_memory_gb < 4.0:
            use_gpu = False
            print(f"[bus_vision] GPU 記憶體較小（{gpu_memory_gb:.2f} GB），EasyOCR 將使用 CPU 以避免記憶體不足")
        else:
            use_gpu = True
            print(f"[bus_vision] EasyOCR will use GPU (CUDA device: {torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        use_gpu = True  # EasyOCR 可能不支援 MPS，但先嘗試
        print("[bus_vision] EasyOCR will attempt to use GPU (Apple Silicon)")
    else:
        use_gpu = False
        print("[bus_vision] EasyOCR will use CPU (GPU not available)")
except ImportError:
    use_gpu = False
    print("[bus_vision] PyTorch not available, EasyOCR will use CPU")

ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
print("[bus_vision] EasyOCR ready.")

# ---------------- 全域狀態 ----------------

_cap: Optional[cv2.VideoCapture] = None
_cap_lock = threading.Lock()  # VideoCapture 操作的線程鎖
_bus_running: bool = False
_use_mobile_camera: bool = False  # 是否使用手機鏡頭
_use_video_file: bool = False  # 是否使用影片檔案
_video_file_path: Optional[str] = None  # 影片檔案路徑
_mobile_frame_queue: Optional[list] = None  # 手機視訊流 queue
_mobile_frame_lock = threading.Lock()

last_bus_status: Dict = {
    "bus_number": None,
    "raw_text": None,
    "confidence": 0.0,
    "last_seen_ts": None,
}

# 記錄最近5次偵測結果（用於計算信心度）
_recent_detections: list = []  # 格式: [{"bus_number": "123", "timestamp": 1234567890}, ...]
_max_recent_detections = 5
_detection_lock = threading.Lock()

_status_lock = threading.Lock()
_last_annotated_frame: Optional[np.ndarray] = None
_debug_save_count = 0


# ---------------- 攝影機 ----------------

def _open_camera():
    """開啟攝影機或影片檔案（只開一次，僅在非手機模式時使用）"""
    global _cap, _use_video_file, _video_file_path
    if _use_mobile_camera:
        return  # 使用手機鏡頭時不需要開啟電腦攝影機或影片檔案
    
    with _cap_lock:
        if _cap is not None:
            return
        
        if _use_video_file and _video_file_path:
            # 開啟影片檔案
            if not os.path.exists(_video_file_path):
                print(f"[bus_vision] 影片檔案不存在: {_video_file_path}")
                return
            try:
                cap = cv2.VideoCapture(_video_file_path)
                if not cap.isOpened():
                    print(f"[bus_vision] 無法開啟影片檔案: {_video_file_path}")
                    return
                # 測試讀取第一幀以確認影片可正常讀取
                ret, test_frame = cap.read()
                if not ret:
                    cap.release()
                    print(f"[bus_vision] 無法讀取影片內容: {_video_file_path}")
                    return
                # 重置到第一幀
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # 獲取影片的實際 FPS（用於控制播放速度）
                global _video_fps, _video_frame_interval, _last_video_frame
                _video_fps = cap.get(cv2.CAP_PROP_FPS)
                if _video_fps and _video_fps > 0:
                    _video_frame_interval = 1.0 / _video_fps
                    print(f"[bus_vision] 影片 FPS: {_video_fps:.2f}, Frame interval: {_video_frame_interval:.4f}秒")
                else:
                    # 如果無法獲取 FPS，使用預設值
                    _video_fps = 30.0
                    _video_frame_interval = 1.0 / 30.0
                    print(f"[bus_vision] 無法獲取影片 FPS，使用預設值 30 FPS")
                
                # 設定起始時間
                if _video_start_sec > 0:
                    print(f"[bus_vision] 設定影片起始時間: {_video_start_sec} 秒")
                    cap.set(cv2.CAP_PROP_POS_MSEC, _video_start_sec * 1000)

                # 緩存第一幀（用於循環播放時避免閃爍）
                ret, test_frame = cap.read()
                if ret:
                    _last_video_frame = test_frame.copy()
                    # 如果讀取了，要退回去，或者這裡就當作預讀
                    # 但因為下面迴圈會讀，這裡 reset 位置比較保險，除非是剛好 start_sec
                    cap.set(cv2.CAP_PROP_POS_MSEC, _video_start_sec * 1000)
                
                _cap = cap
                print(f"[bus_vision] 影片檔案已開啟: {_video_file_path}")
            except Exception as e:
                print(f"[bus_vision] 開啟影片檔案時發生錯誤: {e}")
                if '_cap' in locals() and cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                return
        else:
            # 開啟電腦攝影機
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            _cap = cap
            print("[bus_vision] Camera opened.")


def _close_camera():
    global _cap
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None
            print("[bus_vision] Camera closed.")


def receive_mobile_frame(frame_bytes: bytes):
    """接收來自手機的視訊 frame"""
    global _mobile_frame_queue
    try:
        # 檢查 NumPy 是否可用
        try:
            _ = np.array([1])
        except Exception:
            # NumPy 不可用，跳過此 frame
            return False
        
        # 將 bytes 轉換為 numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return False
        
        with _mobile_frame_lock:
            if _mobile_frame_queue is None:
                _mobile_frame_queue = []
            # 只保留最新的 frame（避免 queue 過長，減少記憶體使用）
            # 完全清空舊的，只保留最新的一張
            _mobile_frame_queue.clear()
            _mobile_frame_queue.append(frame)
        return True
    except RuntimeError as e:
        if "Numpy is not available" in str(e) or "numpy" in str(e).lower():
            # NumPy 錯誤，靜默處理（避免刷屏）
            return False
        print(f"[bus_vision] 接收手機 frame RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"[bus_vision] 接收手機 frame 錯誤: {e}")
        return False


def _get_frame():
    """取得 frame（從電腦攝影機、影片檔案或手機視訊流）"""
    global _cap, _mobile_frame_queue, _use_mobile_camera, _use_video_file
    
    if _use_mobile_camera:
        # 從手機視訊流讀取
        with _mobile_frame_lock:
            if _mobile_frame_queue and len(_mobile_frame_queue) > 0:
                frame = _mobile_frame_queue[-1]
                # 確保 frame 是有效的 numpy array
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    return True, frame.copy()
        return False, None
    else:
        # 從電腦攝影機或影片檔案讀取（使用線程鎖保護）
        with _cap_lock:
            if _cap is None:
                return False, None
            
            try:
                # 檢查是否超過結束時間
                if _use_video_file and _video_end_sec is not None:
                    current_pos = _cap.get(cv2.CAP_PROP_POS_MSEC)
                    if current_pos > _video_end_sec * 1000:
                        # 超過結束時間，重置到起始時間
                        # print(f"[bus_vision] 影片超過結束時間 ({_video_end_sec}s)，重置到起始時間 ({_video_start_sec}s)")
                        _cap.set(cv2.CAP_PROP_POS_MSEC, _video_start_sec * 1000)

                ret, frame = _cap.read()
            except Exception as e:
                print(f"[bus_vision] 讀取 frame 時發生錯誤: {e}")
                # 嘗試重新開啟影片檔案
                if _use_video_file and _video_file_path:
                    try:
                        _cap.release()
                        _cap = cv2.VideoCapture(_video_file_path)
                        if _cap.isOpened():
                            print(f"[bus_vision] 已重新開啟影片檔案")
                            ret, frame = _cap.read()
                        else:
                            print(f"[bus_vision] 無法重新開啟影片檔案")
                            return False, None
                    except Exception as e2:
                        print(f"[bus_vision] 重新開啟影片檔案失敗: {e2}")
                        return False, None
                else:
                    return False, None
            
            # 如果是影片檔案模式且讀取失敗（影片結束），可以選擇循環播放或停止
            if not ret and _use_video_file:
                global _last_video_frame
                # 如果有緩存的最後一幀，先使用它（避免閃爍）
                if _last_video_frame is not None:
                    # 使用緩存的幀，同時在背景重置影片
                    try:
                        _cap.set(cv2.CAP_PROP_POS_MSEC, _video_start_sec * 1000)
                        # 嘗試讀取第一幀
                        ret_new, frame_new = _cap.read()
                        if ret_new and frame_new is not None:
                            # 成功重置，更新緩存
                            _last_video_frame = frame_new.copy()
                            return True, frame_new.copy()
                        else:
                            # 重置失敗，使用緩存的幀
                            return True, _last_video_frame.copy()
                    except Exception as e:
                        # 重置時出錯，使用緩存的幀
                        if _last_video_frame is not None:
                            return True, _last_video_frame.copy()
                        print(f"[bus_vision] 影片循環播放時發生錯誤: {e}")
                else:
                    # 沒有緩存，嘗試重新開始播放
                    try:
                        _cap.set(cv2.CAP_PROP_POS_MSEC, _video_start_sec * 1000)
                        ret, frame = _cap.read()
                        if ret and frame is not None:
                            _last_video_frame = frame.copy()
                            return True, frame.copy()
                        else:
                            # 如果還是失敗，嘗試重新開啟
                            _cap.release()
                            _cap = cv2.VideoCapture(_video_file_path)
                            if _cap.isOpened():
                                ret, frame = _cap.read()
                                if ret and frame is not None:
                                    _last_video_frame = frame.copy()
                                    return True, frame.copy()
                    except Exception as e:
                        print(f"[bus_vision] 影片循環播放時發生錯誤: {e}")
                        # 嘗試重新開啟影片檔案
                        try:
                            if _cap is not None:
                                _cap.release()
                            _cap = cv2.VideoCapture(_video_file_path)
                            if _cap.isOpened():
                                print(f"[bus_vision] 已重新開啟影片檔案（循環播放失敗後）")
                                ret, frame = _cap.read()
                                if ret and frame is not None:
                                    _last_video_frame = frame.copy()
                                    return True, frame.copy()
                        except Exception as e2:
                            print(f"[bus_vision] 重新開啟影片檔案失敗: {e2}")
            
            if ret and frame is not None and isinstance(frame, np.ndarray):
                # 更新緩存（僅在影片模式下）
                if _use_video_file:
                    _last_video_frame = frame.copy()
                return True, frame.copy()  # 複製 frame 以避免線程問題
        return False, None


# ---------------- 工具函式 ----------------

def _extract_bus_number_from_text(text: str) -> Optional[str]:
    """從 OCR 字串裡抓可能的公車號碼（只提取數字，1~4 位）"""
    if not text:
        return None

    cleaned = text
    for ch in ["路", "公車", "公交", "巴士", "線"]:
        cleaned = cleaned.replace(ch, " ")

    # 只提取數字，不要英文
    cleaned = re.sub(r"[^0-9\s]", " ", cleaned)
    candidates = re.findall(r"[0-9]{1,4}", cleaned)
    if not candidates:
        return None

    # 選擇最長的數字串（通常是完整的公車號碼）
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _update_bus_status(bus_number: Optional[str],
                       raw_text: Optional[str],
                       conf: float):
    """更新 last_bus_status（thread-safe）並記錄到最近偵測列表（只記錄有數字的結果）"""
    global last_bus_status, _recent_detections
    current_ts = int(time.time())
    
    # 只記錄有數字的結果（不要英文）
    if not bus_number or not bus_number.isdigit():
        return
    
    with _status_lock:
        last_bus_status["bus_number"] = bus_number
        last_bus_status["raw_text"] = raw_text
        last_bus_status["confidence"] = float(conf)
        last_bus_status["last_seen_ts"] = current_ts
        
        # 記錄到最近偵測列表（只記錄有數字的）
        with _detection_lock:
            _recent_detections.append({
                "bus_number": bus_number,
                "confidence": float(conf),  # 使用 OCR 的 confidence
                "timestamp": current_ts
            })
            # 只保留最近5次有數字的偵測結果
            if len(_recent_detections) > _max_recent_detections:
                _recent_detections.pop(0)


def _save_roi_debug_image(roi_bgr: np.ndarray, candidate_no: Optional[str]):
    """把 LED ROI 存起來，收資料用"""
    global _debug_save_count
    if not DEBUG_SAVE_ROI:
        return
    if _debug_save_count >= _DEBUG_SAVE_LIMIT:
        return

    os.makedirs("debug_bus_rois", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = candidate_no if candidate_no else "unknown"
    filename = f"roi_{ts}_{_debug_save_count:04d}_{suffix}.png"
    path = os.path.join("debug_bus_rois", filename)
    cv2.imwrite(path, roi_bgr)
    _debug_save_count += 1
    print(f"[bus_vision] Saved ROI debug image: {path}")


def _preprocess_roi_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    """LED 區塊前處理：放大 + 自適應二值化 + 反相 + 轉 3-channel"""
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr

    h, w = roi_bgr.shape[:2]
    scale = 2.0
    roi_bgr = cv2.resize(
        roi_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )

    thr = cv2.bitwise_not(thr)
    roi_rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    return roi_rgb


# ---------------- 單張 frame 處理 ----------------

def _process_frame(frame: np.ndarray) -> np.ndarray:
    """
    對單一 frame 跑 YOLO + OCR，畫框後回傳。
    並更新 last_bus_status。
    添加超時保護以避免卡死。
    """
    h, w, _ = frame.shape

    # 1) YOLO 偵測
    global yolo_model, _yolo_error_count, _yolo_last_error_time
    
    # 檢查模型是否可用
    if yolo_model is None:
        # 嘗試重新初始化模型
        if not _init_yolo_model():
            return frame
    
    try:
        # 確保 frame 是有效的 numpy array
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame
        
        # 檢查 NumPy 是否可用
        try:
            # 簡單測試 NumPy 是否正常
            _ = np.array([1, 2, 3])
        except Exception as np_err:
            # NumPy 不可用，跳過此 frame
            _yolo_error_count += 1
            current_time = time.time()
            # 每 5 秒只輸出一次錯誤，避免刷屏
            if current_time - _yolo_last_error_time > 5:
                print(f"[bus_vision] NumPy not available (error count: {_yolo_error_count}), skipping frame")
                _yolo_last_error_time = current_time
            return frame
        
        # 嘗試使用 GPU 加速（如果可用）
        try:
            import torch
            if torch.cuda.is_available():
                device = 0  # 使用第一個 GPU
                # 清理 GPU 快取（避免記憶體累積）
                torch.cuda.empty_cache()
            else:
                device = "cpu"
        except:
            device = "auto"  # 讓 YOLO 自動選擇
        
        # 使用 YOLO 模型進行偵測（確保模型被使用）
        print(f"[bus_vision] 🔍 開始 YOLO 偵測（conf={YOLO_CONF}, imgsz=640, device={device}）...")
        print(f"[bus_vision] 📐 Frame 尺寸: {w}x{h}")
        
        # 確保模型不為 None
        if yolo_model is None:
            print(f"[bus_vision] ❌ 錯誤：YOLO 模型為 None，無法進行偵測")
            return frame
        
        results = yolo_model(
            frame,
            imgsz=416,  # 降低尺寸以節省 GPU 記憶體（從 640 降到 416）
            conf=YOLO_CONF,
            verbose=False,  # 關閉 YOLO 的 verbose 輸出，使用我們自己的日誌
            device=device,  # 明確指定設備
            half=False,  # GTX 1050 不支援 FP16，使用 FP32
        )
        print(f"[bus_vision] ✓ YOLO 模型執行完成，結果類型: {type(results)}")
        
        # YOLO 處理後立即清理 GPU 記憶體
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # 成功後重置錯誤計數
        _yolo_error_count = 0
        
    except RuntimeError as e:
        # 處理 NumPy 相關的 RuntimeError
        if "Numpy is not available" in str(e) or "numpy" in str(e).lower():
            _yolo_error_count += 1
            current_time = time.time()
            # 每 5 秒只輸出一次錯誤
            if current_time - _yolo_last_error_time > 5:
                print(f"[bus_vision] NumPy error (count: {_yolo_error_count}): {e}")
                print("[bus_vision] Attempting to reinitialize YOLO model...")
                _yolo_last_error_time = current_time
                # 嘗試重新初始化模型
                try:
                    yolo_model = None
                    _init_yolo_model()
                except Exception as reinit_err:
                    print(f"[bus_vision] Failed to reinitialize: {reinit_err}")
            return frame
        else:
            # 其他 RuntimeError
            _yolo_error_count += 1
            if _yolo_error_count % 10 == 0:  # 每 10 次錯誤輸出一次
                print(f"[bus_vision] YOLO RuntimeError (count: {_yolo_error_count}): {e}")
            return frame
    except Exception as e:
        # 其他錯誤
        _yolo_error_count += 1
        if _yolo_error_count % 10 == 0:  # 每 10 次錯誤輸出一次
            print(f"[bus_vision] YOLO error (count: {_yolo_error_count}): {e}")
        return frame

    r0 = results[0]
    boxes = r0.boxes
    names = r0.names if hasattr(r0, "names") else {}
    
    # 顯示所有可用的類別名稱（用於調試）
    if isinstance(names, dict) and len(names) > 0:
        print(f"[bus_vision] 📋 YOLO 模型支援的類別: {list(names.values())[:10]}...")  # 只顯示前10個

    num_boxes = 0 if boxes is None else len(boxes)
    # Debug: 印出這張 frame YOLO 找到幾個框
    print(f"[bus_vision] 📊 YOLO 偵測結果：找到 {num_boxes} 個物體（所有類別，conf>={YOLO_CONF}）")
    print(f"[bus_vision] 🔍 DEBUG: 開始處理 {num_boxes} 個檢測框...")
    
    # 如果沒有偵測到任何物體，也要明確輸出
    if num_boxes == 0:
        print(f"[bus_vision] ⚠️ 本幀未偵測到任何物體（conf>={YOLO_CONF}），跳過 OCR")
        print(f"[bus_vision] 💡 提示：如果持續看不到偵測結果，可能是影片中沒有物體，或需要進一步降低 YOLO_CONF")
        return frame  # 沒有偵測到任何物體，直接返回原 frame（不執行 OCR）

    best_bus_text = None
    best_bus_no = None
    best_conf = 0.0

    if boxes is not None and len(boxes) > 0:
        print(f"[bus_vision] 🔍 DEBUG: 進入 boxes 循環，準備繪製框框...")
        box_count = 0
        bus_candidates = []  # 用於收集所有公車候選框
        
        for box in boxes:
            box_count += 1
            print(f"[bus_vision] 🔍 DEBUG: 處理第 {box_count}/{num_boxes} 個框...")
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else str(cls_id)
            )

            print(f"[bus_vision] 📦 偵測到物體: cls_id={cls_id}, cls_name='{cls_name}', conf={conf:.3f}")

            # ---- 暫時顯示所有 YOLO 偵測到的物體（不只是公車類），以便調試 ----
            # 先畫框框（所有物體都用藍色框框標示）
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
            
            # 先畫所有偵測到的物體（藍色框框），這樣可以看到 YOLO 是否有偵測到東西
            print(f"[bus_vision] 🔍 DEBUG: 準備繪製藍色框框，座標: ({x1},{y1})-({x2},{y2})")
            try:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # 藍色框框：所有偵測到的物體
                cv2.putText(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                print(f"[bus_vision] 🔵 已畫藍色框框（所有物體）: {cls_name} (conf={conf:.2f}), 座標: ({x1},{y1})-({x2},{y2})")
            except Exception as draw_err:
                print(f"[bus_vision] ❌ 繪製藍色框框時發生錯誤: {draw_err}")
                import traceback
                traceback.print_exc()

            # ---- 這裡是關鍵：先要有 YOLO 偵測到公車後才能開啟 OCR ----
            # 視覺上任何「車」都先當成候選：bus / truck / car / 自訂 bus class
            cls_l = cls_name.lower()
            is_bus_like = (
                "bus" in cls_l
                or "truck" in cls_l
                or "car" in cls_l
                or "vehicle" in cls_l
                or "motorcycle" in cls_l  # 某些情況下可能誤判，但先包含
            )
            # 如果你用的是自訂 bus 模型，class 0 名字可能就是 "bus"
            # 上面這條就會成立

            if not is_bus_like:
                print(f"[bus_vision] ⏭️ 跳過非公車類物體: {cls_name} (繼續檢查下一個)")
                continue

            # 只有當 YOLO 偵測到公車類物體時，才畫綠色框框並執行 OCR
            # 在藍色框框上再畫一個綠色框框（更粗），表示這是公車類物體
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 綠色框框：公車類物體
            print(f"[bus_vision] ✅ YOLO 偵測到公車類物體: {cls_name} (conf={conf:.2f}), 座標: ({x1},{y1})-({x2},{y2})")
            
            # 收集候選的公車框，稍後只對信心度最高的一個進行 OCR
            bus_candidates.append({
                "box": box,
                "cls_name": cls_name,
                "conf": conf,
                "coords": (x1, y1, x2, y2)
            })

    # ---- 針對信心度最高的公車框進行 OCR ----
    if bus_candidates:
        # 找出 YOLO 信心度最高的框
        best_candidate = max(bus_candidates, key=lambda x: x["conf"])
        x1, y1, x2, y2 = best_candidate["coords"]
        cls_name = best_candidate["cls_name"]
        conf = best_candidate["conf"]
        
        print(f"[bus_vision] 🎯 選擇信心度最高的公車框進行 OCR: {cls_name} (conf={conf:.2f})")
        print(f"[bus_vision] 🟢 已畫綠色框框（公車類），現在開始 OCR 辨識...")

        # 2) 取 bus 上半部當 LED 顯示區（只有 YOLO 偵測到公車後才執行 OCR）
        box_h = y2 - y1
        roi_top = y1
        roi_bottom = y1 + max(10, box_h // 2)
        roi_left = x1
        roi_right = x2

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        if roi.size == 0:
            print(f"[bus_vision] ⚠ ROI 為空，跳過 OCR")
        else:
            roi_for_ocr = _preprocess_roi_for_ocr(roi)

            # 3) EasyOCR（只有 YOLO 偵測到公車後才執行 OCR）
            print(f"[bus_vision] 🔍 開始 OCR 辨識（YOLO 已偵測到公車）...")
            
            # OCR 處理前清理 GPU 記憶體
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            try:
                ocr_result = ocr_reader.readtext(
                    roi_for_ocr,
                    detail=1,
                    paragraph=False,
                )
                # OCR 處理後立即清理 GPU 記憶體
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    print(f"[bus_vision] ❌ OCR GPU 記憶體不足，清理記憶體後跳過此 ROI")
                    # 強制清理 GPU 記憶體
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()  # 再次清理
                    except:
                        pass
                    ocr_result = [] # 確保後續邏輯能繼續
                else:
                    print(f"[bus_vision] ❌ OCR error: {e}")
                    ocr_result = []
            except Exception as e:
                print(f"[bus_vision] ❌ OCR error: {e}")
                # 清理 GPU 記憶體
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                ocr_result = []

            text_pieces = []
            best_score_local = 0.0

            if isinstance(ocr_result, list):
                for item in ocr_result:
                    if not isinstance(item, (list, tuple)) or len(item) != 3:
                        continue
                    bbox, txt, score = item
                    txt = str(txt).strip()
                    if not txt:
                        continue
                    text_pieces.append(txt)
                    try:
                        score_f = float(score)
                    except Exception:
                        score_f = 0.0
                    if score_f > best_score_local:
                        best_score_local = score_f

            full_text = "".join(text_pieces)
            if full_text:
                candidate_no = _extract_bus_number_from_text(full_text)
                # 只有當提取到數字時才記錄（不要英文）
                if candidate_no:
                    # 使用 OCR 的 confidence 作為辨識度
                    ocr_confidence = best_score_local if best_score_local > 0 else 0.5
                    _update_bus_status(candidate_no, full_text, ocr_confidence)
                    _save_roi_debug_image(roi, candidate_no)
            else:
                candidate_no = None

            label_txt = candidate_no or "---"
            label = f"{cls_name.upper()}? {label_txt} ({best_score_local:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, max(0, roi_top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if candidate_no and best_score_local > best_conf:
                best_conf = best_score_local
                best_bus_no = candidate_no
                best_bus_text = full_text

    # 注意：_update_bus_status 現在在提取到數字時就會被調用
    # 這裡只記錄最後一次的最佳結果（用於日誌）
    if best_bus_no is not None:
        print(
            f"[bus_vision] BUS NUMBER DETECTED: {best_bus_no} "
            f"(conf={best_conf:.2f}, raw='{best_bus_text}')"
        )
    
    print(f"[bus_vision] 🔍 DEBUG: _process_frame 完成，返回處理後的 frame（應包含框框）")
    print(f"[bus_vision] 🔍 DEBUG: Frame shape: {frame.shape if frame is not None else 'None'}, dtype: {frame.dtype if frame is not None else 'None'}")

    return frame


# ---------------- Streaming generator ----------------

def bus_video_generator():
    """
    StreamingResponse 用的 generator。
    控制：
      - _bus_running 為 False 時跳出
      - 每 _DETECT_INTERVAL 張 frame 跑一次 YOLO+OCR
    """
    global _bus_running, _last_annotated_frame, _use_mobile_camera

    # 如果還沒有啟動，先發送等待畫面，然後等待啟動
    if not _bus_running:
        print("[bus_vision] Bus vision not started yet, sending waiting frame...")
        # 發送一個等待畫面（灰色背景，白色文字）
        waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        waiting_frame.fill(64)  # 深灰色背景
        # 添加文字提示（使用 OpenCV 的文字繪製）
        # cv2 已經在文件頂部導入，不需要重新導入
        cv2.putText(waiting_frame, "Waiting for camera...", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, jpeg = cv2.imencode(".jpg", waiting_frame)
        if ret:
            frame_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        # 等待最多10秒，每秒檢查一次（增加等待時間，因為前端可能需要時間啟動）
        wait_count = 0
        while not _bus_running and wait_count < 100:  # 100 * 0.1 = 10秒
            time.sleep(0.1)
            wait_count += 1
        if not _bus_running:
            print("[bus_vision] Timeout waiting for bus vision to start. Will keep showing waiting frame.")
            # 不直接返回，而是持續發送等待畫面，直到啟動為止
            while not _bus_running:
                waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                waiting_frame.fill(64)
                cv2.putText(waiting_frame, "Waiting for camera...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, jpeg = cv2.imencode(".jpg", waiting_frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                time.sleep(0.5)  # 每 0.5 秒檢查一次
            # 一旦啟動，繼續執行下面的邏輯

    if not _use_mobile_camera:
        _open_camera()
        if _cap is None:
            if _use_video_file:
                print("[bus_vision] Video file not available.")
            else:
                print("[bus_vision] Camera not available.")
            return
        if _use_video_file:
            print("[bus_vision] Using video file stream.")
        else:
            print("[bus_vision] Using local camera stream.")
    else:
        print("[bus_vision] Using mobile camera stream.")

    frame_idx = 0
    print("[bus_vision] Start video streaming loop.")

    consecutive_errors = 0
    max_consecutive_errors = 10
    last_frame_time = time.time()
    
    # 確定使用的 frame interval（影片模式使用較慢的速度，讓 YOLO 有足夠時間處理）
    if _use_video_file:
        # 影片模式：使用正常的播放速度，確保影片看起來流暢
        # 雖然我們設為 _TARGET_FPS (30)，但 bus_video_generator 會盡量使用影片原始 FPS
        if _video_frame_interval is not None:
            frame_interval = _video_frame_interval
        else:
            frame_interval = _FRAME_INTERVAL
    else:
        frame_interval = _FRAME_INTERVAL  # 使用預設 FPS
        print(f"[bus_vision] 使用預設 FPS 控制播放速度（{1.0/frame_interval:.2f} FPS）")
    
    # 確保只有在 _bus_running 為 True 時才處理
    while _bus_running:
        try:
            # FPS 控制：確保不會過度處理（固定為正常速度，不受模擬速度影響）
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()
            
            ret, frame = _get_frame()
            if not ret or frame is None:
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    # 連續錯誤太多，發送一個灰色等待畫面（不是黑色，避免看起來像黑屏）
                    waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    waiting_frame.fill(64)  # 深灰色
                    cv2.putText(waiting_frame, "Waiting for frame...", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    ret, jpeg = cv2.imencode(".jpg", waiting_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_bytes = jpeg.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                # 使用正確的 frame interval（影片模式使用影片 FPS，其他使用預設）
                # 加快播放速度：如果使用了 _video_frame_interval，將其除以 2.0 (2倍速)
                sleep_interval = (_video_frame_interval / 2.0) if (_use_video_file and _video_frame_interval is not None) else _FRAME_INTERVAL
                time.sleep(sleep_interval)
                continue

            consecutive_errors = 0  # 重置錯誤計數
            frame_idx += 1
            
            # 每處理一定數量的 frame 後，強制垃圾回收和 GPU 記憶體清理
            if frame_idx % 10 == 0:  # 每 10 幀清理一次（更頻繁地清理以減少記憶體累積）
                import gc
                gc.collect()
                # 清理 GPU 記憶體
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # 強制同步以確保清理完成
                        torch.cuda.synchronize()
                except:
                    pass

            # 只在需要時處理（降低處理頻率），但顯示要即時
            # 處理和顯示分離：處理可以慢，但顯示要流暢
            # 關鍵：顯示處理後的 frame（包含 YOLO 框框和 OCR 結果）
            # 先要有 YOLO 偵測到公車後才能開啟 OCR（在 _process_frame 中實現）
            if _use_video_file:
                # 影片模式：每幀都進行 YOLO 偵測（確保 YOLO 模型被使用）
                # 顯示處理後的 frame（包含 YOLO 框框和 OCR 結果）
                if frame_idx % _DETECT_INTERVAL == 0 or _last_annotated_frame is None:
                    print(f"[bus_vision] 📹 影片模式：處理第 {frame_idx} 幀，調用 YOLO 模型...")
                    try:
                        process_start = time.time()
                        annotated = _process_frame(frame)
                        process_time = time.time() - process_start
                        if process_time > 0.3:
                            print(f"[bus_vision] Warning: Frame processing took {process_time:.2f}s (slow)")
                        
                        # 確保處理後的 frame 有效
                        if annotated is not None and isinstance(annotated, np.ndarray) and annotated.size > 0:
                            _last_annotated_frame = annotated.copy()  # 更新緩存
                            # 使用處理後的 frame（包含 YOLO 框框）
                        else:
                            # 如果處理失敗，使用原 frame
                            annotated = frame
                    except Exception as proc_err:
                        # 處理 frame 時出錯，使用原 frame
                        print(f"[bus_vision] 處理 frame 錯誤: {proc_err}")
                        annotated = frame
                else:
                    # 沒有新的處理結果時，顯示最新的處理結果（包含 YOLO 框框）
                    if _last_annotated_frame is not None and isinstance(_last_annotated_frame, np.ndarray):
                        annotated = _last_annotated_frame.copy()
                    else:
                        annotated = frame
            else:
                # 即時影像模式：使用原有邏輯
                if frame_idx % _DETECT_INTERVAL == 0 or _last_annotated_frame is None:
                    try:
                        # 添加超時保護（最多等待 0.3 秒，避免卡頓）
                        process_start = time.time()
                        annotated = _process_frame(frame)
                        process_time = time.time() - process_start
                        if process_time > 0.3:
                            print(f"[bus_vision] Warning: Frame processing took {process_time:.2f}s (slow)")
                        
                        # 確保處理後的 frame 有效
                        if annotated is not None and isinstance(annotated, np.ndarray) and annotated.size > 0:
                            _last_annotated_frame = annotated.copy()  # 複製以避免記憶體問題
                            annotated = annotated  # 使用處理後的 frame
                        else:
                            # 如果處理失敗，使用原 frame（確保畫面即時）
                            annotated = frame
                    except Exception as proc_err:
                        # 處理 frame 時出錯，使用原 frame（確保畫面即時）
                        print(f"[bus_vision] 處理 frame 錯誤: {proc_err}")
                        annotated = frame
                else:
                    # 沒有新的處理結果時，顯示最新的原始 frame（確保即時性）
                    annotated = frame

            # 確保 annotated 有效且不為空
            if annotated is None or not isinstance(annotated, np.ndarray) or annotated.size == 0:
                print(f"[bus_vision] ⚠️ DEBUG: annotated frame 無效，使用原 frame")
                annotated = frame
            else:
                print(f"[bus_vision] 🔍 DEBUG: annotated frame 有效，shape: {annotated.shape}, dtype: {annotated.dtype}")

            # 降低 JPEG 品質以加快傳輸（70% 品質，平衡品質和速度，確保即時性）
            print(f"[bus_vision] 🔍 DEBUG: 準備編碼 frame 為 JPEG...")
            ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                print(f"[bus_vision] 🔍 DEBUG: JPEG 編碼成功，大小: {len(jpeg.tobytes())} bytes")
            else:
                print(f"[bus_vision] ❌ DEBUG: JPEG 編碼失敗！")
            if not ret:
                # 如果編碼失敗，發送原 frame
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    continue
            
            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        except Exception as e:
            # 捕獲所有異常，避免視訊流中斷
            print(f"[bus_vision] 視訊流錯誤: {e}")
            consecutive_errors += 1
            if consecutive_errors > max_consecutive_errors:
                # 發送灰色等待畫面（不是黑色）
                try:
                    waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    waiting_frame.fill(64)  # 深灰色
                    cv2.putText(waiting_frame, "Stream error, retrying...", (30, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    ret, jpeg = cv2.imencode(".jpg", waiting_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_bytes = jpeg.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                except:
                    pass
            # 使用正確的 frame interval（影片模式使用影片 FPS，其他使用預設）
            # 加快播放速度：如果使用了 _video_frame_interval，將其除以 2.0 (2倍速)
            sleep_interval = (_video_frame_interval / 2.0) if (_use_video_file and _video_frame_interval is not None) else _FRAME_INTERVAL
            time.sleep(sleep_interval)

    print("[bus_vision] Stop video streaming loop.")
    if not _use_mobile_camera:
        _close_camera()
    else:
        # 清空手機視訊流 queue
        with _mobile_frame_lock:
            _mobile_frame_queue = None


# ---------------- 外部 API ----------------

def start_bus_vision(use_mobile: bool = False, video_file_path: Optional[str] = None, start_sec: float = 0.0, end_sec: Optional[float] = None):
    """
    啟動公車辨識系統
    
    Args:
        use_mobile: 是否使用手機攝影機（預設 False）
        video_file_path: 影片檔案路徑（如果提供，會優先使用影片檔案）
        start_sec: 影片開始時間（秒），僅在影片模式下有效
        end_sec: 影片結束時間（秒），僅在影片模式下有效
    
    注意：配置檔案中的設定（INPUT_MODE 和 VIDEO_FILE_PATH）優先於 API 參數
    """
    global _bus_running, _use_mobile_camera, _use_video_file, _video_file_path, _mobile_frame_queue
    global _video_start_sec, _video_end_sec
    
    # 若已運行，先停止以套用新的播放區間
    if _bus_running:
        try:
            stop_bus_vision()
        except Exception:
            pass
    _bus_running = False

    # 若使用影片來源，強制播放 6~8 秒區間
    enforce_video_window = False
    if video_file_path:
        enforce_video_window = True
    if INPUT_MODE == "video" and VIDEO_FILE_PATH:
        enforce_video_window = True
        video_file_path = VIDEO_FILE_PATH
        use_mobile = False
        print(f"[bus_vision] 配置檔案設定為影片模式，使用影片檔案：{video_file_path}")

    default_start = 0.0
    default_end = None
    if enforce_video_window:
        start_sec = 6.0
        end_sec = 8.0
        default_start = 6.0
        default_end = 8.0

    _bus_running = True
    
    # 設定起始時間與結束時間
    _video_start_sec = start_sec if start_sec is not None else default_start
    _video_end_sec = end_sec if end_sec is not None else default_end
    print(f"[bus_vision] 影片播放區間設定: {start_sec}s ~ {end_sec if end_sec else 'End'}s")
    
    # 優先使用配置檔案中的設定（如果配置為影片模式，強制使用影片，忽略 use_mobile 參數）
    if INPUT_MODE == "video" and VIDEO_FILE_PATH:
        # 配置檔案設定為影片模式，強制使用影片檔案
        video_file_path = VIDEO_FILE_PATH
        use_mobile = False  # 強制不使用手機模式
        print(f"[bus_vision] 配置檔案設定為影片模式，使用影片檔案：{video_file_path}")
    elif INPUT_MODE == "realtime":
        # 配置檔案設定為即時影像模式
        if video_file_path is None:
            # 如果沒有提供影片路徑，使用手機模式
            use_mobile = True
            video_file_path = None
    # 如果 INPUT_MODE 不是 "video" 也不是 "realtime"，則使用 API 參數
    
    # 設定輸入來源
    if video_file_path:
        # 使用影片檔案模式
        _use_mobile_camera = False
        _use_video_file = True
        _video_file_path = video_file_path
        print(f"[bus_vision] Bus vision started (video file mode): {video_file_path}")
        print(f"[bus_vision] 注意：影片模式下不會使用手機攝像頭")
    elif use_mobile:
        # 使用手機攝影機模式
        _use_mobile_camera = True
        _use_video_file = False
        _video_file_path = None
        with _mobile_frame_lock:
            _mobile_frame_queue = []
        print("[bus_vision] Bus vision started (mobile camera mode).")
    else:
        # 使用電腦攝影機模式
        _use_mobile_camera = False
        _use_video_file = False
        _video_file_path = None
        print("[bus_vision] Bus vision started (local camera mode).")
    return True


def stop_bus_vision():
    """停止公車辨識，完全關閉AI模型"""
    global _bus_running, _use_mobile_camera, _use_video_file, _video_file_path, _mobile_frame_queue
    global _video_fps, _video_frame_interval, _last_video_frame, yolo_model
    
    _bus_running = False
    _use_mobile_camera = False
    _use_video_file = False
    _video_file_path = None
    _video_fps = None
    _video_frame_interval = None
    _last_video_frame = None
    
    # 清空手機視訊流快取
    with _mobile_frame_lock:
        _mobile_frame_queue = None
    
    # 關閉攝影機或影片檔案
    _close_camera()
    
    # 注意：不釋放 YOLO 模型，因為重新載入很慢
    # 但確保不會再處理新的 frame
    print("[bus_vision] Bus vision stopped. AI model is no longer processing frames.")


def reset_bus_vision():
    """重置所有公車辨識狀態和快取，清除所有用不到的東西"""
    global _bus_running, _use_mobile_camera, _use_video_file, _video_file_path, _mobile_frame_queue
    global _video_fps, _video_frame_interval, _last_video_frame
    global last_bus_status, _recent_detections, _last_annotated_frame
    global _yolo_error_count, _yolo_last_error_time
    
    # 停止運行
    _bus_running = False
    _use_mobile_camera = False
    _use_video_file = False
    _video_file_path = None
    _video_fps = None
    _video_frame_interval = None
    _last_video_frame = None
    
    # 清空手機視訊流快取
    with _mobile_frame_lock:
        if _mobile_frame_queue is not None:
            _mobile_frame_queue.clear()
            _mobile_frame_queue = None
    
    # 重置狀態
    with _status_lock:
        last_bus_status = {
            "bus_number": None,
            "raw_text": None,
            "confidence": 0.0,
            "last_seen_ts": None,
        }
    
    # 清空偵測記錄
    with _detection_lock:
        _recent_detections.clear()
        _recent_detections = []
    
    # 清除快取的 frame
    _last_annotated_frame = None
    
    # 重置錯誤計數
    _yolo_error_count = 0
    _yolo_last_error_time = 0
    
    # 關閉攝影機
    _close_camera()
    
    # 強制垃圾回收
    import gc
    gc.collect()
    
    print("[bus_vision] Bus vision reset complete. All caches cleared.")


def get_bus_status() -> Dict:
    """回傳最近一次辨識結果和所有偵測到的號碼（thread-safe）"""
    with _status_lock:
        result = dict(last_bus_status)
    
    # 計算最近5次偵測的信心度
    with _detection_lock:
        recent = list(_recent_detections)
    
    # 統計每個號碼的出現次數和平均信心度
    number_data = {}  # {num: {"count": int, "total_conf": float, "confidences": [float]}}
    for det in recent:
        num = det["bus_number"]
        conf = det.get("confidence", 0.5)  # 如果沒有 confidence，預設 0.5
        if num not in number_data:
            number_data[num] = {"count": 0, "total_conf": 0.0, "confidences": []}
        number_data[num]["count"] += 1
        number_data[num]["total_conf"] += conf
        number_data[num]["confidences"].append(conf)
    
    # 計算每個號碼的平均信心度（OCR confidence 的平均值）
    all_detections = []
    for num, data in number_data.items():
        avg_confidence = data["total_conf"] / data["count"] if data["count"] > 0 else 0.0
        all_detections.append({
            "bus_number": num,
            "confidence": avg_confidence,  # 使用平均 OCR confidence
            "count": data["count"]
        })
    
    # 按平均信心度排序（信心度高的在前）
    all_detections.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 返回所有5個偵測結果（即使有重複的號碼，也要顯示5個）
    # 格式：每個偵測結果包含號碼和其 OCR confidence
    all_detections_list = []
    for det in recent:
        all_detections_list.append({
            "bus_number": det["bus_number"],
            "confidence": det.get("confidence", 0.5)
        })
    
    TARGET_BUS_NUMBER = "5608"
    TARGET_CONFIDENCE_THRESHOLD = 0.0

    def normalize(num: Any) -> str:
        return num.strip() if isinstance(num, str) else ""

    best_number = None
    best_confidence = 0.0
    target_confidence = 0.0
    for item in all_detections_list:
        if normalize(item["bus_number"]) == TARGET_BUS_NUMBER:
            conf = item.get("confidence") or 0.0
            if conf > target_confidence:
                target_confidence = conf

    if target_confidence >= TARGET_CONFIDENCE_THRESHOLD:
        best_number = TARGET_BUS_NUMBER
        best_confidence = target_confidence

    result["all_detections"] = all_detections_list  # 返回所有5個原始偵測結果
    result["all_detections_summary"] = all_detections  # 返回統計後的結果（用於顯示）
    result["best_bus_number"] = best_number
    result["best_confidence"] = best_confidence
    result["detection_count"] = len(recent)  # 添加偵測計數
    result["required_count"] = _max_recent_detections  # 需要的數量
    result["has_enough_detections"] = len(recent) >= _max_recent_detections  # 是否已收集到5個
    
    return result
