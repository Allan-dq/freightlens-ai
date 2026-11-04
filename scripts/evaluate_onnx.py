"""Export the trained detector to ONNX (640x640, no NMS) and score the export on the test split.

The browser demo runs this file, so its accuracy is measured separately from the PyTorch
checkpoint. Writes metrics/pallet_detection_onnx.json under the config's artifacts root.
"""

import argparse
import json

from freightlens.config import load_config
from freightlens.io import write_json
from freightlens.vision.detector import dataset_yaml, load_detector, trained_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    size = config.vision.detector.image_size
    exported = load_detector(trained_checkpoint(config)).export(
        format="onnx", imgsz=size, opset=17, simplify=True, dynamic=False, nms=False, device="cpu"
    )
    from ultralytics import YOLO

    model = YOLO(exported, task="detect")
    results = model.val(
        data=str(dataset_yaml(config).resolve()),
        split="test",
        imgsz=size,
        batch=1,
        device="cpu",
        plots=False,
        verbose=False,
        project=str((config.paths.artifacts / "figures").resolve()),
        name="onnx_eval",
        exist_ok=True,
    )
    names = {int(k): v for k, v in model.names.items()}
    per_class = {
        names[int(c)]: {"AP50": float(results.box.ap50[i]), "AP50_95": float(results.box.ap[i])}
        for i, c in enumerate(results.box.ap_class_index)
    }
    metrics = {
        "model": f"{exported} ({size}x{size} square input, browser demo)",
        "split": "test",
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "pallet_AP50": per_class["pallet"]["AP50"],
        "pallet_AP50_95": per_class["pallet"]["AP50_95"],
        "per_class": per_class,
    }
    write_json(config.paths.artifacts / "metrics/pallet_detection_onnx.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
