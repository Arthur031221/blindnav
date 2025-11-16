# app/bus_vision.py

import os
import cv2
import time
import threading
import re
from typing import Dict, Optional

import numpy as np
from ultralytics import YOLO
import easyocr

# ---------------- 設定區 ----------------

# YOLO 權重：如果你有自己的 bus 模型，可以改這裡，例如 "bus_best.pt"
YOLO_WEIGHTS = "yolov8n.pt"

# YOLO conf 門檻：先設低一點，確保抓得到東西
YOLO_CONF = 0.15

# 幀率控制：每幾張 frame 做一次 YOLO+OCR
_DETECT_INTERVAL = 5  # 5 張做一次偵測（降低處理頻率，提升流暢度）

# 視訊流 FPS 限制（避免過度處理導致卡頓）
_TARGET_FPS = 30  # 目標 30 FPS，流暢且即時（顯示用，處理頻率由 _DETECT_INTERVAL 控制）
_FRAME_INTERVAL = 1.0 / _TARGET_FPS  # 每幀間隔時間（秒）

# Debug：是否把 bus LED ROI 存成圖片
DEBUG_SAVE_ROI = True
_DEBUG_SAVE_LIMIT = 200

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
# 嘗試使用 GPU（如果可用）
try:
    import torch
    if torch.cuda.is_available():
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
_bus_running: bool = False
_use_mobile_camera: bool = False  # 是否使用手機鏡頭
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
    """開啟攝影機（只開一次，僅在非手機模式時使用）"""
    global _cap
    if _use_mobile_camera:
        return  # 使用手機鏡頭時不需要開啟電腦攝影機
    if _cap is not None:
        return
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _cap = cap
    print("[bus_vision] Camera opened.")


def _close_camera():
    global _cap
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
    """取得 frame（從電腦攝影機或手機視訊流）"""
    global _cap, _mobile_frame_queue, _use_mobile_camera
    
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
        # 從電腦攝影機讀取
        if _cap is None:
            return False, None
        ret, frame = _cap.read()
        if ret and frame is not None and isinstance(frame, np.ndarray):
            return True, frame
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
        
        # 使用較小的圖片尺寸以加快處理速度（GPU 記憶體有限）
        results = yolo_model(
            frame,
            imgsz=416,  # 從 640 降到 416，加快處理速度
            conf=YOLO_CONF,
            verbose=False,
            device=device,  # 明確指定設備
            half=False,  # GTX 1050 不支援 FP16，使用 FP32
        )
        
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

    num_boxes = 0 if boxes is None else len(boxes)
    # Debug: 印出這張 frame YOLO 找到幾個框
    print(f"[bus_vision] YOLO detected {num_boxes} boxes.")

    best_bus_text = None
    best_bus_no = None
    best_conf = 0.0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = (
                names.get(cls_id, str(cls_id))
                if isinstance(names, dict)
                else str(cls_id)
            )

            print(f"[bus_vision] box: cls={cls_id} ({cls_name}), conf={conf:.2f}")

            # ---- 這裡是關鍵：不要只限定 bus=5 ----
            # 視覺上任何「車」都先當成候選：bus / truck / car / 自訂 bus class
            cls_l = cls_name.lower()
            is_bus_like = (
                "bus" in cls_l
                or "truck" in cls_l
                or "car" in cls_l
            )
            # 如果你用的是自訂 bus 模型，class 0 名字可能就是 "bus"
            # 上面這條就會成立

            if not is_bus_like:
                continue

            num_boxes += 1

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2) 取 bus 上半部當 LED 顯示區
            box_h = y2 - y1
            roi_top = y1
            roi_bottom = y1 + max(10, box_h // 2)
            roi_left = x1
            roi_right = x2

            roi = frame[roi_top:roi_bottom, roi_left:roi_right]
            if roi.size == 0:
                continue

            roi_for_ocr = _preprocess_roi_for_ocr(roi)

            # 3) EasyOCR
            try:
                ocr_result = ocr_reader.readtext(
                    roi_for_ocr,
                    detail=1,
                    paragraph=False,
                )
            except Exception as e:
                print("[bus_vision] OCR error:", e)
                continue

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
            print("[bus_vision] Camera not available.")
            return
    else:
        print("[bus_vision] Using mobile camera stream.")

    frame_idx = 0
    print("[bus_vision] Start video streaming loop.")

    consecutive_errors = 0
    max_consecutive_errors = 10
    last_frame_time = time.time()
    
    # 確保只有在 _bus_running 為 True 時才處理
    while _bus_running:
        try:
            # FPS 控制：確保不會過度處理
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < _FRAME_INTERVAL:
                time.sleep(_FRAME_INTERVAL - elapsed)
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
                time.sleep(_FRAME_INTERVAL)  # 使用統一的幀間隔
                continue

            consecutive_errors = 0  # 重置錯誤計數
            frame_idx += 1
            
            # 每處理一定數量的 frame 後，強制垃圾回收和 GPU 記憶體清理
            if frame_idx % 30 == 0:  # 每 30 幀清理一次
                import gc
                gc.collect()
                # 清理 GPU 記憶體
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

            # 只在需要時處理（降低處理頻率），但顯示要即時
            # 處理和顯示分離：處理可以慢，但顯示要流暢
            # 關鍵：即使沒有新的處理結果，也要顯示最新的原始 frame（確保即時性）
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
                # 這樣可以保證畫面流暢，不會卡在舊的處理結果上
                annotated = frame
                
                # 如果有舊的處理結果，可以選擇性地疊加（但為了即時性，優先顯示原 frame）
                # 如果需要顯示處理結果，可以取消下面的註釋：
                # if _last_annotated_frame is not None and isinstance(_last_annotated_frame, np.ndarray):
                #     annotated = _last_annotated_frame.copy()

            # 確保 annotated 有效且不為空
            if annotated is None or not isinstance(annotated, np.ndarray) or annotated.size == 0:
                annotated = frame

            # 降低 JPEG 品質以加快傳輸（70% 品質，平衡品質和速度，確保即時性）
            ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
            time.sleep(_FRAME_INTERVAL)

    print("[bus_vision] Stop video streaming loop.")
    if not _use_mobile_camera:
        _close_camera()
    else:
        # 清空手機視訊流 queue
        with _mobile_frame_lock:
            _mobile_frame_queue = None


# ---------------- 外部 API ----------------

def start_bus_vision(use_mobile: bool = False):
    global _bus_running, _use_mobile_camera, _mobile_frame_queue
    if _bus_running:
        return False
    _bus_running = True
    _use_mobile_camera = use_mobile
    if use_mobile:
        with _mobile_frame_lock:
            _mobile_frame_queue = []
        print("[bus_vision] Bus vision started (mobile camera mode).")
    else:
        print("[bus_vision] Bus vision started (local camera mode).")
    return True


def stop_bus_vision():
    """停止公車辨識，完全關閉AI模型"""
    global _bus_running, _use_mobile_camera, _mobile_frame_queue
    global yolo_model
    
    _bus_running = False
    _use_mobile_camera = False
    
    # 清空手機視訊流快取
    with _mobile_frame_lock:
        _mobile_frame_queue = None
    
    # 關閉攝影機
    _close_camera()
    
    # 注意：不釋放 YOLO 模型，因為重新載入很慢
    # 但確保不會再處理新的 frame
    print("[bus_vision] Bus vision stopped. AI model is no longer processing frames.")


def reset_bus_vision():
    """重置所有公車辨識狀態和快取，清除所有用不到的東西"""
    global _bus_running, _use_mobile_camera, _mobile_frame_queue
    global last_bus_status, _recent_detections, _last_annotated_frame
    global _yolo_error_count, _yolo_last_error_time
    
    # 停止運行
    _bus_running = False
    _use_mobile_camera = False
    
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
    
    # 最有信心的號碼（平均 OCR confidence 最高的）
    # 當收集到至少5個偵測結果時，只有信心度高於90%才選出最有信心的
    # 並且號碼必須是有效的（不能是"0"或空字串，長度至少1位）
    best_number = None
    best_confidence = 0.0
    CONFIDENCE_THRESHOLD = 0.90  # 90% 信心度門檻
    
    # 過濾掉無效的號碼（"0"或空字串）
    def is_valid_bus_number(num):
        if not num or not isinstance(num, str):
            return False
        # 移除空白
        num = num.strip()
        # 不能是"0"或空字串，且必須是純數字
        if num == "0" or num == "" or not num.isdigit():
            return False
        # 長度至少1位，最多4位
        if len(num) < 1 or len(num) > 4:
            return False
        return True
    
    if len(recent) >= _max_recent_detections:
        if len(all_detections) > 0:
            # 從統計結果中選出信心度最高的，但必須高於90%且號碼有效
            for candidate in all_detections:
                if candidate["confidence"] >= CONFIDENCE_THRESHOLD and is_valid_bus_number(candidate["bus_number"]):
                    best_number = candidate["bus_number"]
                    best_confidence = candidate["confidence"]
                    break
        elif len(all_detections_list) > 0:
            # 如果統計結果為空，直接從原始結果中選信心度最高的且有效的
            valid_items = [item for item in all_detections_list if is_valid_bus_number(item["bus_number"])]
            if valid_items:
                best_item = max(valid_items, key=lambda x: x["confidence"])
                if best_item["confidence"] >= CONFIDENCE_THRESHOLD:
                    best_number = best_item["bus_number"]
                    best_confidence = best_item["confidence"]
    
    result["all_detections"] = all_detections_list  # 返回所有5個原始偵測結果
    result["all_detections_summary"] = all_detections  # 返回統計後的結果（用於顯示）
    result["best_bus_number"] = best_number
    result["best_confidence"] = best_confidence
    result["detection_count"] = len(recent)  # 添加偵測計數
    result["required_count"] = _max_recent_detections  # 需要的數量
    result["has_enough_detections"] = len(recent) >= _max_recent_detections  # 是否已收集到5個
    
    return result
