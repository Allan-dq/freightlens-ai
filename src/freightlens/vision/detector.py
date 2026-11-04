import os
import shutil
from pathlib import Path

from freightlens.config import Config
from freightlens.data.seeds import set_global_seed
from freightlens.devices import resolve_device
from freightlens.io import write_json


def dataset_yaml(config: Config) -> Path:
    name = config.vision.detector.dataset
    path = config.paths.data / f"processed/{name}/dataset.yaml"
    if not path.exists():
        raise ValueError(f"Prepare the {name} dataset before training or evaluation")
    return path


def load_detector(model: str):
    settings_root = Path.cwd() / ".cache/ultralytics"
    settings_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_root))
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO requires: uv sync --extra vision") from error
    return YOLO(model)


def pallet_class(model) -> int:
    classes = [int(key) for key, name in model.names.items() if name.lower() == "pallet"]
    if len(classes) != 1:
        raise ValueError(
            "Checkpoint has no unique pallet class. COCO pretrained weights cannot "
            "be treated as a pallet detector; fine-tune first."
        )
    return classes[0]


def trained_checkpoint(config: Config) -> str:
    checkpoint = config.paths.models / "vision/pallet/weights/best.pt"
    return str(checkpoint) if checkpoint.exists() else config.vision.detector.model


def detect(path: Path, config: Config) -> list[dict]:
    model = load_detector(trained_checkpoint(config))
    class_id = pallet_class(model)
    results = model.predict(
        str(path),
        classes=[class_id],
        device=resolve_device(config.vision.detector.device),
        verbose=False,
        imgsz=config.vision.detector.image_size,
        conf=config.vision.tracker.confidence_threshold,
    )
    return [
        {
            "bbox_xyxy": box.xyxy[0].tolist(),
            "confidence": float(box.conf[0]),
            "object_class": "pallet",
        }
        for result in results
        for box in result.boxes
    ]


def train_detector(config: Config) -> dict:
    dataset = dataset_yaml(config)
    model = load_detector(config.vision.detector.model)
    set_global_seed(config.seed)
    settings = config.vision.detector
    result = model.train(
        data=str(dataset.resolve()),
        imgsz=settings.image_size,
        epochs=settings.epochs,
        batch=settings.batch_size,
        patience=settings.patience,
        device=resolve_device(settings.device),
        workers=settings.workers,
        seed=config.seed,
        deterministic=True,
        project=str((config.paths.models / "vision").resolve()),
        name="pallet",
        exist_ok=True,
        pretrained=True,
        plots=True,
        verbose=False,
        val=settings.validate_each_epoch,
    )
    weights = config.paths.models / "vision/pallet/weights"
    if not settings.validate_each_epoch:
        shutil.copyfile(weights / "last.pt", weights / "best.pt")
    metrics = {
        "status": "trained",
        "source": settings.model,
        "dataset": settings.dataset,
        "device": resolve_device(settings.device),
        "epochs_requested": settings.epochs,
        "checkpoint_rule": "best validation fitness"
        if settings.validate_each_epoch
        else "last epoch",
        "metrics": result.results_dict,
    }
    write_json(config.paths.artifacts / "metrics/detector_training.json", metrics)
    from freightlens.vision.metrics import evaluate_detector

    return evaluate_detector(config)
