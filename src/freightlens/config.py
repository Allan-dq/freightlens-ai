"""Validated configuration with recursive, relative YAML inheritance."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Paths(StrictModel):
    data: Path = Path("data")
    models: Path = Path("models")
    artifacts: Path = Path("artifacts")


class Sizes(StrictModel):
    train_size: int = Field(200, ge=0)
    val_size: int = Field(60, ge=0)
    test_size: int = Field(60, ge=0)


class PallesetConfig(Sizes):
    max_download_mb: int = Field(1800, gt=0)


class OCRConfig(StrictModel):
    backend: Literal["paddleocr", "fixture"] = "paddleocr"


class LayoutConfig(StrictModel):
    enabled: bool = True
    model_name: str = "microsoft/layoutlmv3-base"
    epochs: int = Field(5, gt=0)
    batch_size: int = Field(2, gt=0)
    patience: int = Field(2, gt=0)
    freeze_backbone: bool = True


class DocumentConfig(StrictModel):
    synthetic: Sizes = Field(default_factory=Sizes)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    layoutlm: LayoutConfig = Field(default_factory=LayoutConfig)


class DetectorConfig(StrictModel):
    model: str = "yolo11n.pt"
    image_size: int = Field(640, ge=32)
    epochs: int = Field(30, gt=0)
    batch_size: int = Field(4, gt=0)
    patience: int = Field(5, gt=0)


class TrackerConfig(StrictModel):
    type: Literal["bytetrack", "fixture"] = "bytetrack"
    confidence_threshold: float = Field(0.25, ge=0, le=1)
    direction: Literal[-1, 0, 1] = 1


class VideoConfig(StrictModel):
    count: int = Field(20, gt=0)
    seconds: int = Field(10, ge=1, le=20)
    fps: int = Field(12, ge=1, le=30)
    width: int = Field(960, ge=160, le=1280)
    height: int = Field(540, ge=120, le=720)


class VisionConfig(StrictModel):
    palleset: PallesetConfig = Field(default_factory=PallesetConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)


class ReconciliationConfig(StrictModel):
    method: Literal["rules"] = "rules"
    scenarios: int = Field(2000, gt=0)
    document_threshold: float = Field(0.7, ge=0, le=1)
    vision_threshold: float = Field(0.6, ge=0, le=1)
    fragmentation_threshold: float = Field(0.4, ge=0, le=1)


class APIConfig(StrictModel):
    max_upload_mb: int = Field(20, gt=0)
    max_video_seconds: int = Field(30, gt=0)
    max_video_pixels: int = Field(2073600, gt=0)


class Config(StrictModel):
    seed: int = Field(42, ge=0)
    mode: Literal["smoke", "full"] = "full"
    paths: Paths = Field(default_factory=Paths)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    api: APIConfig = Field(default_factory=APIConfig)


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    def read(current: Path, visited: set[Path]) -> dict[str, Any]:
        current = current.resolve()
        if current in visited:
            raise ValueError(f"Cyclic config inheritance: {current}")
        visited.add(current)
        raw = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config must be a YAML mapping")
        parent = raw.pop("extends", None)
        return merge(read(current.parent / parent, visited), raw) if parent else raw

    return Config.model_validate(read(Path(path), set()))
