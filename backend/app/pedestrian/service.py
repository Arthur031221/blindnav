from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .detector import SignalDetector
from .video_processor import VideoProcessor

settings = get_settings()
live_preview_path = settings.data_dir / "live_preview.jpg"
live_preview_path.parent.mkdir(parents=True, exist_ok=True)

detector = SignalDetector(settings.model_path)
video_processor = VideoProcessor(detector)

processing_status: Dict[str, Any] = {
    "active": False,
    "current_frame": 0,
    "total_frames": 0,
    "current_image": None,
    "current_state": "unknown",
    "progress": 0,
    "preview_version": 0,
    "meta": {},
}


async def start_processing():
    """Kick off pedestrian-signal detection for the default demo video."""
    if processing_status["active"]:
        raise HTTPException(status_code=409, detail="行人號誌辨識正在執行中")

    input_path = settings.default_video_path
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="找不到預設影片檔案")

    output_name = f"pedestrian_{int(time.time())}.mp4"
    output_path = settings.processed_dir / output_name

    processing_status.update(
        {
            "active": True,
            "current_frame": 0,
            "total_frames": 0,
            "progress": 0,
            "current_state": "unknown",
            "current_image": None,
            "preview_version": int(time.time() * 1000),
            "meta": {
                "video_path": str(input_path),
                "start_sec": 15.0,
                "end_sec": 27.0,
                "frame_interval_sec": 0.5,
            },
        }
    )

    def update_progress(current: int, total: int, state: str, img_base64: str):
        processing_status["current_frame"] = current
        processing_status["total_frames"] = total
        processing_status["current_state"] = state
        processing_status["current_image"] = f"data:image/jpeg;base64,{img_base64}"
        live_preview_path.write_bytes(base64.b64decode(img_base64))
        processing_status["preview_version"] = int(time.time() * 1000)
        processing_status["progress"] = int((current / total) * 100) if total else 0

    try:
        start_exec = time.time()
        summary = await run_in_threadpool(
            video_processor.process_segment,
            input_path,
            output_path,
            15.0,
            27.0,
            conf=0.35,
            iou=0.45,
            progress_callback=update_progress,
        )
        summary.processing_seconds = round(time.time() - start_exec, 2)
        return {
            "success": True,
            "processed_video": output_name,
            "summary": summary.to_dict(),
        }
    finally:
        processing_status["active"] = False


def get_progress() -> Dict[str, Any]:
    return processing_status


def get_processed_video_response(filename: str) -> FileResponse:
    video_path = settings.processed_dir / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="找不到處理後的影片")
    return FileResponse(video_path, media_type="video/mp4")


def get_live_preview_response() -> FileResponse:
    if not live_preview_path.exists():
        raise HTTPException(status_code=404, detail="尚未產生即時影像")
    return FileResponse(
        live_preview_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )

