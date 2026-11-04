"""Choose the pallet-counting confidence threshold on validation data.

AP integrates over all confidence thresholds, but counting needs one operating point.
Predictions are cached once at a low threshold; the count error is then evaluated for a
grid of thresholds on the validation split, the threshold with the lowest mean absolute
count error is selected, and the test split is scored once at that threshold.
"""

from pathlib import Path

import numpy as np

from freightlens.config import Config
from freightlens.devices import resolve_device
from freightlens.io import record_run, write_json

GRID = np.round(np.arange(0.05, 0.81, 0.01), 2)
FLOOR = 0.01


def count_errors(confidences: list[np.ndarray], truths: np.ndarray, threshold: float):
    counts = np.array([int((c >= threshold).sum()) for c in confidences])
    errors = counts - truths
    return float(np.abs(errors).mean()), float(errors.mean())


def select_threshold(confidences: list[np.ndarray], truths: np.ndarray, grid=GRID) -> dict:
    """Return the grid threshold with the lowest mean absolute count error (ties: lowest)."""
    curve = [(float(t), *count_errors(confidences, truths, float(t))) for t in grid]
    best = min(curve, key=lambda row: (row[1], row[0]))
    return {
        "threshold": best[0],
        "mae": best[1],
        "bias": best[2],
        "curve": [{"threshold": t, "mae": m, "bias": b} for t, m, b in curve],
    }


def _predict_split(model, root: Path, split: str, class_id: int, config: Config):
    images = sorted(
        p for p in (root / "images" / split).iterdir() if p.suffix.lower() in {".jpg", ".png"}
    )
    confidences, truths = [], []
    device = resolve_device(config.vision.detector.device)
    for path in images:
        rows = (root / "labels" / split / f"{path.stem}.txt").read_text().splitlines()
        truths.append(sum(1 for r in rows if r.split() and int(r.split()[0]) == class_id))
        result = model.predict(
            str(path),
            classes=[class_id],
            conf=FLOOR,
            iou=0.7,
            max_det=300,
            imgsz=config.vision.detector.image_size,
            device=device,
            verbose=False,
        )[0]
        confidences.append(result.boxes.conf.cpu().numpy())
    return confidences, np.array(truths)


def calibrate_count_threshold(config: Config) -> dict:
    from freightlens.vision.detector import (
        dataset_yaml,
        load_detector,
        pallet_class,
        trained_checkpoint,
    )

    model = load_detector(trained_checkpoint(config))
    class_id = pallet_class(model)
    root = dataset_yaml(config).parent
    has_validation = (
        bool(config.vision.loco.val_subsets) or config.vision.detector.dataset != "loco"
    )
    if has_validation:
        val_conf, val_truth = _predict_split(model, root, "val", class_id, config)
        choice = select_threshold(val_conf, val_truth)
    else:
        # Final fixed-epoch model: no validation split is left, so keep the configured
        # threshold, which was selected on validation for the development model.
        val_truth = np.zeros(0, dtype=int)
        choice = {
            "threshold": config.vision.tracker.confidence_threshold,
            "mae": None,
            "bias": None,
            "curve": [],
        }
    test_conf, test_truth = _predict_split(model, root, "test", class_id, config)
    baseline = config.vision.tracker.confidence_threshold
    test_mae, test_bias = count_errors(test_conf, test_truth, choice["threshold"])
    base_mae, base_bias = count_errors(test_conf, test_truth, baseline)
    result = {
        "status": "measured",
        "selected_on": "validation"
        if has_validation
        else "configured (validation of the development model)",
        "threshold": choice["threshold"],
        "val_mae": choice["mae"],
        "val_bias": choice["bias"],
        "test_mae": test_mae,
        "test_bias": test_bias,
        "baseline_threshold": baseline,
        "test_mae_at_baseline": base_mae,
        "test_bias_at_baseline": base_bias,
        "test_mean_pallets_per_image": float(test_truth.mean()),
        "val_images": len(val_truth),
        "test_images": len(test_truth),
        "val_curve": choice["curve"],
    }
    write_json(config.paths.artifacts / "metrics/pallet_count_calibration.json", result)
    if choice["curve"]:
        _plot(choice["curve"], result, config.paths.artifacts / "figures/count_threshold.png")
    record_run(config, "count_calibration", {k: v for k, v in result.items() if k != "val_curve"})
    return {k: v for k, v in result.items() if k != "val_curve"}


def _plot(curve: list[dict], result: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in curve]
    figure, axis = plt.subplots(figsize=(6, 3.4))
    axis.plot(thresholds, [row["mae"] for row in curve], color="#16794c", label="validation MAE")
    axis.axvline(result["threshold"], color="#16794c", linestyle="--", linewidth=1)
    axis.axvline(result["baseline_threshold"], color="#9a9a94", linestyle=":", linewidth=1)
    axis.set(xlabel="Confidence threshold", ylabel="Pallets per image (MAE)")
    axis.set_title("Pallet count error vs. threshold (LOCO validation session)", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
