import numpy as np

from freightlens.config import Config
from freightlens.io import record_run, write_json


def iou(a, b) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / max(union, 1e-12))


def evaluate_detector(config: Config) -> dict:
    from freightlens.vision.detector import load_detector, pallet_class, trained_checkpoint

    model = load_detector(trained_checkpoint(config))
    class_id = pallet_class(model)
    dataset = config.paths.data / "processed/palleset/dataset.yaml"
    if not dataset.exists():
        raise ValueError("Palleset is not prepared")
    settings = config.vision.detector
    results = model.val(
        data=str(dataset.resolve()),
        split="test",
        device="cpu",
        workers=0,
        imgsz=settings.image_size,
        batch=settings.batch_size,
        project=str((config.paths.artifacts / "figures").resolve()),
        name="detector_eval",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    count_errors, examples = [], []
    root = config.paths.data / "processed/palleset"
    images = sorted((root / "images/test").glob("*.png"))
    for path in images:
        truth_count = len((root / "labels/test" / f"{path.stem}.txt").read_text().split()) // 5
        prediction = model.predict(
            str(path),
            classes=[class_id],
            device="cpu",
            verbose=False,
            imgsz=settings.image_size,
            conf=config.vision.tracker.confidence_threshold,
        )[0]
        count_errors.append(abs(len(prediction.boxes) - truth_count))
        examples.append(
            {"image": str(path), "expected": truth_count, "predicted": len(prediction.boxes)}
        )
        destination = config.paths.artifacts / "figures/detection_examples" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        prediction.save(filename=str(destination))
    metrics = {
        "status": "measured",
        "evaluation_scope": "public_synthetic_palleset_test",
        "checkpoint": trained_checkpoint(config),
        "samples": len(images),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "count_mae": float(np.mean(count_errors)),
        "examples": examples,
    }
    write_json(config.paths.artifacts / "metrics/pallet_detection.json", metrics)
    write_json(config.paths.artifacts / "metrics/pallet_detection_model_probe.json", metrics)
    record_run(config, "detector", metrics)
    return metrics


def match_boxes(predictions, truths, threshold: float = 0.3):
    from scipy.optimize import linear_sum_assignment

    if not predictions or not truths:
        return []
    similarities = np.array([[iou(p, t) for t in truths] for p in predictions])
    rows, columns = linear_sum_assignment(1 - similarities)
    return [
        (int(p), int(t))
        for p, t in zip(rows, columns, strict=True)
        if similarities[p, t] >= threshold
    ]
