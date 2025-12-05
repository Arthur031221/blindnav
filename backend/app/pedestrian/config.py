from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class PedestrianSettings:
    """
    Configuration used by the pedestrian traffic-light integration.
    Values default to the original project directories but can be
    overridden through environment variables if needed.
    """

    root_path: Path
    model_path: Path
    default_video_path: Path
    data_dir: Path
    processed_dir: Path


@lru_cache
def get_settings() -> PedestrianSettings:
    # Allow overriding via environment variables; fall back to the original repo paths.
    root_fallback = Path(os.getenv("PEDESTRIAN_PROJECT_ROOT", r"D:\traffic light"))
    model_path = Path(
        os.getenv(
            "PEDESTRIAN_MODEL_PATH",
            str(root_fallback / "Pedestrian-Traffic-Light-Detection" / "pedestrianlight.pt"),
        )
    )
    default_video = Path(
        os.getenv(
            "PEDESTRIAN_VIDEO_PATH",
            str(root_fallback / "data" / "input" / "IMG_3372.MOV"),
        )
    )

    data_dir = Path(os.getenv("PEDESTRIAN_DATA_DIR", str(Path("pedestrian_data"))))
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    return PedestrianSettings(
        root_path=root_fallback,
        model_path=model_path,
        default_video_path=default_video,
        data_dir=data_dir,
        processed_dir=processed_dir,
    )

