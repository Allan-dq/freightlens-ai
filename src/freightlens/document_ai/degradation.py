"""Seeded scan corruption; return a homography so labels stay aligned."""

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel


class DegradationConfig(BaseModel):
    level: Literal["clean", "light", "medium", "hard"] = "light"
    rotation_degrees: float = 0
    perspective_pixels: float = 0
    blur_kernel: int = 1
    motion_blur: bool = False
    brightness: float = 0
    contrast: float = 1
    jpeg_quality: int = 100
    noise_fraction: float = 0
    tint: int = 0
    shadow: float = 0
    edge_crop: int = 0
    resolution_scale: float = 1


def degrade(image: np.ndarray, level: str, seed: int):
    rng = np.random.default_rng(seed)
    strength = {"clean": 0, "light": 1, "medium": 2, "hard": 3}[level]
    cfg = DegradationConfig.model_validate({"level": level})
    matrix = np.eye(3)
    if not strength:
        return image.copy(), cfg, matrix
    h, w = image.shape[:2]
    cfg.rotation_degrees = float(rng.uniform(-strength, strength))
    cfg.perspective_pixels = float(strength * 2)
    cfg.blur_kernel = int(rng.choice([1, 3] if strength < 3 else [3, 5]))
    cfg.motion_blur = bool(strength > 1 and rng.random() < 0.4)
    cfg.brightness = float(rng.uniform(-8, 8) * strength)
    cfg.contrast = float(rng.uniform(1 - 0.1 * strength, 1))
    cfg.jpeg_quality = int(rng.integers(85 - strength * 15, 96 - strength * 8))
    cfg.noise_fraction = 0.0003 * strength
    cfg.tint = int(strength * 3)
    cfg.shadow = float(rng.uniform(0, 0.07 * strength))
    cfg.edge_crop = int(rng.integers(0, 5 * strength))
    cfg.resolution_scale = 1 - 0.16 * strength
    rotation = np.vstack(
        [cv2.getRotationMatrix2D((w / 2, h / 2), cfg.rotation_degrees, 1), [0, 0, 1]]
    )
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    offsets = rng.uniform(-cfg.perspective_pixels, cfg.perspective_pixels, (4, 2))
    perspective = cv2.getPerspectiveTransform(corners, (corners + offsets).astype(np.float32))
    matrix = perspective @ rotation
    result = cv2.warpPerspective(image, matrix, (w, h), borderValue=(248, 248, 244))
    result = cv2.GaussianBlur(result, (cfg.blur_kernel, cfg.blur_kernel), 0)
    if cfg.motion_blur:
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = 0.2
        result = cv2.filter2D(result, -1, kernel)
    result = result.astype(np.float32) * cfg.contrast + cfg.brightness
    result[:, :, 2] -= cfg.tint
    result *= np.linspace(1, 1 - cfg.shadow, w)[None, :, None]
    noise = rng.random((h, w))
    result[noise < cfg.noise_fraction / 2] = 0
    result[noise > 1 - cfg.noise_fraction / 2] = 255
    result = np.clip(result, 0, 255).astype(np.uint8)
    if cfg.edge_crop:
        c = cfg.edge_crop
        result[:c] = 255
        result[-c:] = 255
        result[:, :c] = 255
    small = cv2.resize(result, None, fx=cfg.resolution_scale, fy=cfg.resolution_scale)
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), cfg, matrix


def transform_box(box, matrix, width, height):
    x1, y1, x2, y2 = box
    points = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    points = cv2.perspectiveTransform(points, matrix)[0]
    return [
        float(np.clip(points[:, 0].min(), 0, width - 1)),
        float(np.clip(points[:, 1].min(), 0, height - 1)),
        float(np.clip(points[:, 0].max(), 1, width)),
        float(np.clip(points[:, 1].max(), 1, height)),
    ]
