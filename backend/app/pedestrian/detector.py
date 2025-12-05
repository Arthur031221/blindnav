from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
from ultralytics import YOLO

SignalState = Literal["walk", "stop", "unknown"]


@dataclass
class FrameDecision:
    frame_index: int
    timestamp: float
    state: SignalState
    confidences: Dict[SignalState, float]


class SignalDetector:
    """
    Thin wrapper around the YOLO model used in the pedestrian
    traffic light project. Copied here so the BlindNav backend
    can reuse it without running a separate service.
    """

    def __init__(self, model_path: Path, device: Optional[str] = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.label_map = {int(idx): name for idx, name in self.model.names.items()}

    def predict(self, frame, conf: float, iou: float):
        return self.model.predict(
            frame,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
            imgsz=320,
            half=False,
        )[0]

    def summarize_state(self, result, frame_index: int, timestamp: float) -> FrameDecision:
        confidences: Dict[SignalState, float] = {"walk": 0.0, "stop": 0.0, "unknown": 0.0}

        boxes = result.boxes
        if boxes is not None and boxes.cls is not None:
            classes = boxes.cls.tolist()
            scores = boxes.conf.tolist()
            for cls_id, score in zip(classes, scores):
                name = self.label_map.get(int(cls_id))
                if name == "green":
                    confidences["walk"] = max(confidences["walk"], float(score))
                elif name == "red":
                    confidences["stop"] = max(confidences["stop"], float(score))

        if confidences["walk"] == 0 and confidences["stop"] == 0:
            confidences["unknown"] = 1.0

        state: SignalState = "unknown"
        if confidences["walk"] > confidences["stop"]:
            state = "walk"
        elif confidences["stop"] > confidences["walk"]:
            state = "stop"

        return FrameDecision(
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            confidences=confidences,
        )

