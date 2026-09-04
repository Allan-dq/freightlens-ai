"""Real PaddleOCR adapter plus an explicitly selected, ground-truth fixture backend."""

from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from freightlens.io import read_json


class OCRToken(BaseModel):
    text: str
    box: tuple[float, float, float, float]
    confidence: float = Field(ge=0, le=1)
    order: int = 0
    page: int = 0


class OCRResult(BaseModel):
    tokens: list[OCRToken]
    backend: str
    is_fixture: bool = False

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)


class OCRBackend(Protocol):
    def extract(self, image: np.ndarray) -> OCRResult: ...


class FixtureOCRBackend:
    def __init__(self, annotation: Path, corrupt: bool = True):
        self.annotation = read_json(annotation)
        self.corrupt = corrupt

    def extract(self, image: np.ndarray) -> OCRResult:
        annotation = self.annotation
        rng = np.random.default_rng(annotation["seed"] + 197)
        level = annotation["degradation"]["level"]
        rate = {"clean": 0, "light": 0.005, "medium": 0.025, "hard": 0.07}[level]
        tokens = []
        for index, source in enumerate(annotation["tokens"]):
            value = source["text"]
            if self.corrupt and rng.random() < rate:
                if rng.random() < 0.5:
                    continue
                value = value.replace("0", "O").replace("1", "I") if value.isdigit() else value[:-1]
            if value:
                tokens.append(
                    OCRToken(text=value, box=source["box"], order=index, confidence=1 - rate * 4)
                )
        return OCRResult(tokens=tokens, backend="fixture_ground_truth_tokens", is_fixture=True)


class PaddleOCRBackend:
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError("PaddleOCR requires: uv sync --extra ocr") from error
        self.engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )

    def extract(self, image: np.ndarray) -> OCRResult:
        tokens: list[OCRToken] = []
        for result in self.engine.predict(np.ascontiguousarray(image[:, :, ::-1])):
            texts = result["rec_texts"]
            boxes = result["rec_polys"]
            scores = result["rec_scores"]
            for value, polygon, score in zip(texts, boxes, scores, strict=True):
                polygon = np.asarray(polygon)
                box = [
                    float(polygon[:, 0].min()),
                    float(polygon[:, 1].min()),
                    float(polygon[:, 0].max()),
                    float(polygon[:, 1].max()),
                ]
                # Paddle returns line boxes. Split proportionally for spatial parsing.
                cursor = box[0]
                for word in value.split():
                    length = (box[2] - box[0]) * len(word) / max(len(value), 1)
                    tokens.append(
                        OCRToken(
                            text=word,
                            box=(cursor, box[1], cursor + length, box[3]),
                            confidence=float(score),
                            order=len(tokens),
                        )
                    )
                    cursor += length + (box[2] - box[0]) / max(len(value), 1)
        return OCRResult(tokens=tokens, backend="paddleocr")
