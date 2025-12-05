from __future__ import annotations

import base64
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import cv2

from .detector import FrameDecision, SignalDetector, SignalState


class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int, state: str, img_base64: str) -> None: ...


@dataclass
class VideoSummary:
    filename: str
    output_path: str
    frame_count: int
    fps: float
    duration: float
    state_counts: Dict[SignalState, int]
    timeline: List[FrameDecision]
    dominant_state: SignalState
    processing_seconds: float

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "output_path": self.output_path,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration": self.duration,
            "state_counts": self.state_counts,
            "dominant_state": self.dominant_state,
            "processing_seconds": self.processing_seconds,
            "timeline": [
                {
                    "frame_index": item.frame_index,
                    "timestamp": round(item.timestamp, 2),
                    "state": item.state,
                    "confidences": item.confidences,
                }
                for item in self.timeline
            ],
        }


class VideoProcessor:
    def __init__(self, detector: SignalDetector) -> None:
        self.detector = detector

    def process_segment(
        self,
        input_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        conf: float,
        iou: float,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> VideoSummary:
        """
        Process a fixed segment of the video (start_time ~ end_time seconds)
        and emit progress updates for the UI.
        """
        frames_dir = input_path.parent.parent / "frames_temp" / f"{input_path.stem}_{int(time.time())}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError("無法開啟影片")

            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 640)
            frame_size = (width, height)

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            frame_interval = max(int(fps * 0.5), 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_index = start_frame
            extracted_frames = []
            while frame_index < end_frame:
                success, frame = cap.read()
                if not success:
                    break

                if (frame_index - start_frame) % frame_interval == 0:
                    frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(
                        {
                            "path": frame_path,
                            "index": frame_index,
                            "timestamp": frame_index / fps,
                            "frame": frame,
                        }
                    )

                frame_index += 1

            cap.release()

            timeline: List[FrameDecision] = []
            counter: Counter = Counter()

            for i, frame_info in enumerate(extracted_frames):
                frame = cv2.imread(str(frame_info["path"]))
                result = self.detector.predict(frame, conf=conf, iou=iou)
                decision = self.detector.summarize_state(
                    result,
                    frame_info["index"],
                    frame_info["timestamp"],
                )

                annotated = result.plot()
                self._draw_status_overlay(annotated, decision)

                annotated_path = frames_dir / f"annotated_{frame_info['index']:06d}.jpg"
                cv2.imwrite(str(annotated_path), annotated)

                timeline.append(decision)
                counter[decision.state] += 1

                if progress_callback:
                    _, buffer = cv2.imencode(".jpg", annotated)
                    img_base64 = base64.b64encode(buffer).decode("utf-8")
                    progress_callback(i + 1, len(extracted_frames), decision.state, img_base64)

            cap = cv2.VideoCapture(str(input_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

            frame_index = start_frame
            result_idx = 0
            processed_count = 0

            while frame_index < end_frame:
                success, frame = cap.read()
                if not success:
                    break

                if result_idx < len(extracted_frames):
                    if result_idx + 1 < len(extracted_frames):
                        if frame_index < extracted_frames[result_idx + 1]["index"]:
                            current_decision = timeline[result_idx]
                        else:
                            result_idx += 1
                            current_decision = timeline[result_idx]
                    else:
                        current_decision = timeline[result_idx]
                else:
                    current_decision = timeline[-1] if timeline else None

                if current_decision:
                    self._draw_status_overlay(frame, current_decision)

                writer.write(frame)
                processed_count += 1
                frame_index += 1

            cap.release()
            writer.release()

            duration = (end_frame - start_frame) / fps
            dominant: SignalState = "unknown"
            if counter:
                dominant = max(counter.items(), key=lambda item: item[1])[0]

            return VideoSummary(
                filename=input_path.name,
                output_path=output_path.name,
                frame_count=processed_count,
                fps=fps,
                duration=round(duration, 2),
                state_counts=dict(counter),
                timeline=timeline,
                dominant_state=dominant,
                processing_seconds=0.0,
            )

        finally:
            if frames_dir.exists():
                shutil.rmtree(frames_dir)

    @staticmethod
    def _draw_status_overlay(frame, decision: FrameDecision) -> None:
        colors = {
            "walk": (0, 200, 0),
            "stop": (0, 0, 255),
            "unknown": (255, 215, 0),
        }
        labels = {
            "walk": "小綠人：可通行",
            "stop": "小紅人：請停止",
            "unknown": "尚未偵測到行人號誌",
        }

        color = colors.get(decision.state, (255, 255, 255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(
            frame,
            labels.get(decision.state, "未知狀態"),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )
        conf_text = f"Walk:{decision.confidences['walk']:.2f} Stop:{decision.confidences['stop']:.2f}"
        cv2.putText(
            frame,
            conf_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

