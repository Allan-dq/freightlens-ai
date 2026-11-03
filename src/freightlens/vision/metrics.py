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
    from freightlens.devices import resolve_device
    from freightlens.vision.detector import (
        dataset_yaml,
        load_detector,
        pallet_class,
        trained_checkpoint,
    )

    model = load_detector(trained_checkpoint(config))
    class_id = pallet_class(model)
    dataset = dataset_yaml(config)
    settings = config.vision.detector
    device = resolve_device(settings.device)
    results = model.val(
        data=str(dataset.resolve()),
        split="test",
        device=device,
        workers=settings.workers,
        imgsz=settings.image_size,
        batch=settings.batch_size,
        project=str((config.paths.artifacts / "figures").resolve()),
        name="detector_eval",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    count_errors, examples = [], []
    root = dataset.parent
    images = sorted(
        p for p in (root / "images/test").iterdir() if p.suffix.lower() in {".png", ".jpg"}
    )
    for path in images:
        rows = (root / "labels/test" / f"{path.stem}.txt").read_text().splitlines()
        truth_count = sum(1 for row in rows if row.split() and int(row.split()[0]) == class_id)
        prediction = model.predict(
            str(path),
            classes=[class_id],
            device=device,
            verbose=False,
            imgsz=settings.image_size,
            conf=config.vision.tracker.confidence_threshold,
        )[0]
        count_errors.append(abs(len(prediction.boxes) - truth_count))
        examples.append(
            {"image": str(path), "expected": truth_count, "predicted": len(prediction.boxes)}
        )
        if len(examples) <= 24:
            destination = config.paths.artifacts / "figures/detection_examples" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            prediction.save(filename=str(destination))
    names = model.names
    per_class = {
        names[int(c)]: {
            "AP50": float(results.box.ap50[i]),
            "AP50_95": float(results.box.ap[i]),
        }
        for i, c in enumerate(results.box.ap_class_index)
    }
    scope = {
        "palleset": "public_synthetic_palleset_test",
        "loco": "loco_real_warehouse_held_out_sessions",
    }[settings.dataset]
    metrics = {
        "status": "measured",
        "evaluation_scope": scope,
        "dataset": settings.dataset,
        "checkpoint": trained_checkpoint(config),
        "device": device,
        "samples": len(images),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "pallet_AP50": per_class.get("pallet", {}).get("AP50"),
        "pallet_AP50_95": per_class.get("pallet", {}).get("AP50_95"),
        "per_class": per_class,
        "pallet_count_mae": float(np.mean(count_errors)),
        "speed_ms_per_image": {k: float(v) for k, v in results.speed.items()},
        "examples": examples[:50],
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
