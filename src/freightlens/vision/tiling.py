"""Sliced (tiled) inference for small, distant objects, with a shared AP evaluator.

LOCO frames are 1920x1080; at a 640 px input a distant pallet is only a few pixels tall.
Tiled inference runs the same detector on overlapping crops (optionally plus the full
frame), shifts the boxes back to frame coordinates and merges duplicates with class-wise
NMS. No retraining is involved, so it measures how much recall is lost to resolution.

Every mode is scored by the same evaluator: predictions are matched to ground truth with
the greedy IoU matching Ultralytics uses and AP comes from ``ultralytics.utils.metrics``,
so tiled and full-frame numbers are directly comparable.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from freightlens.config import Config
from freightlens.devices import resolve_device
from freightlens.io import record_run, write_json

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
EVAL_CONF = 0.001
MAX_DET = 300


@dataclass(frozen=True)
class Mode:
    name: str
    tile: int | None = None
    overlap: float = 0.2
    full_frame: bool = True
    merge_iou: float = 0.6


DEFAULT_MODES = [
    Mode("full_frame_640"),
    Mode("tiles_960_plus_full", tile=960),
    Mode("tiles_640_plus_full", tile=640),
]


def tile_grid(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Overlapping tile windows (x0, y0, x1, y1) that cover the whole frame."""

    def starts(size: int) -> list[int]:
        if tile >= size:
            return [0]
        step = max(1, int(round(tile * (1 - overlap))))
        positions = list(range(0, size - tile + 1, step))
        if positions[-1] + tile < size:
            positions.append(size - tile)
        return positions

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]


def nms_merge(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou: float):
    """Class-wise NMS over boxes gathered from several crops."""
    if len(boxes) == 0:
        return boxes, scores, classes
    import torch
    from torchvision.ops import batched_nms

    keep = batched_nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(scores).float(),
        torch.from_numpy(classes).long(),
        iou,
    ).numpy()[:MAX_DET]
    return boxes[keep], scores[keep], classes[keep]


def _to_arrays(result, dx: int = 0, dy: int = 0):
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
    boxes[:, [0, 2]] += dx
    boxes[:, [1, 3]] += dy
    return boxes, result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)


def predict(model, image_bgr: np.ndarray, mode: Mode, imgsz: int, device: str):
    """Return merged (boxes xyxy, scores, classes) for one frame in the given mode."""
    height, width = image_bgr.shape[:2]
    common = {
        "imgsz": imgsz,
        "conf": EVAL_CONF,
        "iou": 0.7,
        "max_det": MAX_DET,
        "device": device,
        "verbose": False,
    }
    parts = []
    if mode.full_frame or mode.tile is None:
        parts.append(_to_arrays(model.predict(image_bgr, **common)[0]))
    if mode.tile is not None:
        windows = tile_grid(width, height, mode.tile, mode.overlap)
        crops = [np.ascontiguousarray(image_bgr[y0:y1, x0:x1]) for x0, y0, x1, y1 in windows]
        for (x0, y0, _, _), result in zip(windows, model.predict(crops, **common), strict=True):
            parts.append(_to_arrays(result, x0, y0))
    boxes = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 4))
    scores = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    classes = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, int)
    if mode.tile is None:
        return boxes, scores, classes
    return nms_merge(boxes, scores, classes, mode.merge_iou)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b))."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_boxes, pred_cls, true_boxes, true_cls) -> np.ndarray:
    """Greedy IoU matching identical to Ultralytics' validator: (N, 10) correctness."""
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return correct
    iou = box_iou(true_boxes, pred_boxes) * (true_cls[:, None] == pred_cls[None, :])
    for i, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def read_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [r.split() for r in path.read_text().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, int)
    values = np.array(rows, dtype=float)
    cx, cy, w, h = (
        values[:, 1] * width,
        values[:, 2] * height,
        values[:, 3] * width,
        values[:, 4] * height,
    )
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return boxes, values[:, 0].astype(int)


def evaluate_split(
    config: Config,
    split: str,
    modes: list[Mode] = DEFAULT_MODES,
    count_thresholds: dict[str, float] | None = None,
) -> dict:
    """Score each mode on one split.

    On the validation split the pallet-counting threshold is selected per mode; pass those
    thresholds as ``count_thresholds`` when scoring the test split so it is used only once.
    """
    import cv2
    from ultralytics.utils.metrics import ap_per_class

    from freightlens.vision.calibration import count_errors, select_threshold
    from freightlens.vision.detector import (
        dataset_yaml,
        load_detector,
        pallet_class,
        trained_checkpoint,
    )

    model = load_detector(trained_checkpoint(config))
    pallet = pallet_class(model)
    names = model.names
    root = dataset_yaml(config).parent
    device = resolve_device(config.vision.detector.device)
    imgsz = config.vision.detector.image_size
    images = sorted(
        p for p in (root / "images" / split).iterdir() if p.suffix.lower() in {".jpg", ".png"}
    )
    stats: dict[str, dict[str, list[np.ndarray]]] = {
        m.name: {"tp": [], "conf": [], "cls": [], "pallet_conf": []} for m in modes
    }
    targets, pallet_truth = [], []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        height, width = image.shape[:2]
        true_boxes, true_cls = read_labels(
            root / "labels" / split / f"{path.stem}.txt", width, height
        )
        targets.append(true_cls)
        pallet_truth.append(int((true_cls == pallet).sum()))
        for mode in modes:
            boxes, scores, classes = predict(model, image, mode, imgsz, device)
            entry = stats[mode.name]
            entry["tp"].append(match(boxes, classes, true_boxes, true_cls))
            entry["conf"].append(scores)
            entry["cls"].append(classes)
            entry["pallet_conf"].append(scores[classes == pallet])
    target_cls = np.concatenate(targets)
    truth = np.array(pallet_truth)
    report: dict = {"split": split, "images": len(images), "image_size": imgsz, "modes": {}}
    for mode in modes:
        entry = stats[mode.name]
        result = ap_per_class(
            np.concatenate(entry["tp"]),
            np.concatenate(entry["conf"]),
            np.concatenate(entry["cls"]),
            target_cls,
        )
        ap, classes_present = result[5], result[6]
        per_class = {
            names[int(c)]: {"AP50": float(ap[i, 0]), "AP50_95": float(ap[i].mean())}
            for i, c in enumerate(classes_present)
        }
        if count_thresholds is None:
            counting = select_threshold(entry["pallet_conf"], truth)
            threshold, source = counting["threshold"], "selected on this split"
        else:
            threshold, source = count_thresholds[mode.name], "given (selected on validation)"
        mae, bias = count_errors(entry["pallet_conf"], truth, threshold)
        report["modes"][mode.name] = {
            "tile": mode.tile,
            "overlap": mode.overlap,
            "full_frame": mode.full_frame,
            "mAP50": float(ap[:, 0].mean()),
            "mAP50_95": float(ap.mean()),
            "pallet_AP50": per_class["pallet"]["AP50"],
            "pallet_AP50_95": per_class["pallet"]["AP50_95"],
            "per_class": per_class,
            "count_threshold": threshold,
            "count_threshold_source": source,
            "count_mae": mae,
            "count_bias": bias,
        }
    write_json(config.paths.artifacts / f"metrics/tiling_{split}.json", report)
    summary = {
        name: {k: v for k, v in values.items() if k != "per_class"}
        for name, values in report["modes"].items()
    }
    record_run(config, f"tiling_{split}", summary)
    return summary
