"""Publish trained weights to Hugging Face model repositories with generated model cards.

Cards are rendered from the measured metric files, so published numbers match the README.
Use --dry-run to write the cards to build/model-cards/ without uploading.
"""

import argparse
import json
import shutil
from pathlib import Path

from freightlens.config import load_config

GITHUB = "https://github.com/Allan-dq/freightlens-ai"
DEMO = "https://huggingface.co/spaces/AllanDai/freightlens-ai"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detector_card(final_metrics: Path, dev_metrics: Path, manifest: dict, epochs: int) -> str:
    final = _read(final_metrics / "pallet_detection.json")
    development = _read(dev_metrics / "pallet_detection.json")
    onnx = _read(final_metrics / "pallet_detection_onnx.json")
    calibration = _read(final_metrics / "pallet_count_calibration.json")
    latency = _read(dev_metrics / "latency.json")["detector_yolo11n_640"]
    cpu_ms = latency["cpu"]["median_ms"]
    mps_ms = latency.get("mps", {}).get("median_ms", float("nan"))
    splits = manifest["splits"]
    rows = "\n".join(
        f"| {name.replace('_', ' ')} | {values['AP50']:.3f} | {values['AP50_95']:.3f} |"
        for name, values in sorted(final["per_class"].items())
    )
    train = ", ".join(map(str, splits["train"]["subsets"]))
    test = " and ".join(map(str, splits["test"]["subsets"]))
    return f"""---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags: [yolo11, logistics, warehouse, pallet-detection, onnx]
---

# FreightLens YOLO11n (LOCO)

YOLO11n fine-tuned on [LOCO](https://github.com/tum-fml/loco) (CC0), real photos from logistics
facilities, to detect pallets, small load carriers, stillages, forklifts and pallet trucks.
Part of [FreightLens AI]({GITHUB}); try it in the browser in the [live demo]({DEMO}).

## Files

- `freightlens-yolo11n-loco.pt`: Ultralytics checkpoint.
- `freightlens-yolo11n-loco.onnx`: ONNX export, 640x640 input, no NMS (used by the browser demo).

## Data and training

LOCO is recorded as walks through facilities, so splits are whole recording sessions. A
development model trained on sessions 2 and 3 and used session 5 to choose the epoch count
(best epoch 37). This final model trains on sessions {train} ({splits["train"]["images"]:,}
images) for {epochs} epochs and keeps the last checkpoint. Sessions {test}, LOCO's official
validation sessions ({splits["test"]["images"]:,} images), were never used for training or
selection and were scored once. COCO-pretrained YOLO11n, 640 px input, batch 16, Apple M1 Pro
(MPS).

## Test results (unseen sessions)

| Class | AP50 | AP50-95 |
|---|---|---|
{rows}
| **all classes (mAP)** | **{final["mAP50"]:.3f}** | **{final["mAP50_95"]:.3f}** |

- Development model (sessions 2 and 3 only): pallet AP50 {development["pallet_AP50"]:.3f},
  mAP50 {development["mAP50"]:.3f}. The third session adds {
        final["pallet_AP50"] - development["pallet_AP50"]:+.3f} pallet AP50.
- ONNX export: pallet AP50 {onnx["pallet_AP50"]:.3f} (PyTorch: {final["pallet_AP50"]:.3f}).
- Pallet counting at confidence {calibration["threshold"]:.2f} (selected on validation for the
  development model): mean absolute error {calibration["test_mae"]:.1f} pallets per image, bias
  {calibration["test_bias"]:+.1f}, with {calibration["test_mean_pallets_per_image"]:.0f} pallets
  per image on average.
- Median latency per image (same architecture and input size): {cpu_ms:.0f} ms CPU,
  {mps_ms:.0f} ms MPS (Apple M1 Pro).

## Limitations

Small, distant pallets are often missed at 640 px. Forklifts and pallet trucks have only a few
hundred training boxes. LOCO covers five facilities filmed from a walking camera; fixed dock
cameras, other pallet types and night lighting are not represented, and a counting threshold
chosen at one site does not transfer reliably to another.
"""


def layout_card(metrics: Path) -> str:
    word = _read(metrics / "document_layoutlm_funsd.json")
    segment = _read(metrics / "document_layoutlm_funsd-segment.json")
    per_entity = "\n".join(
        f"| {name.lower()} | {values['precision']:.3f} | {values['recall']:.3f} | "
        f"{values['f1-score']:.3f} |"
        for name, values in sorted(word["per_entity"].items())
    )
    return f"""---
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: token-classification
base_model: microsoft/layoutlmv3-base
tags: [layoutlmv3, funsd, document-ai, form-understanding]
---

# FreightLens LayoutLMv3 (FUNSD)

`microsoft/layoutlmv3-base` fine-tuned end to end on [FUNSD](https://guillaumejaume.github.io/FUNSD/)
scanned forms to tag words as question, answer or header (BIO labels). Part of
[FreightLens AI]({GITHUB}).

## Training

{word["split"]["train"]} training forms, {word["split"]["val"]} held-out forms for checkpoint
selection, official {word["split"]["test"]}-form test split scored once. 1,000 steps, AdamW,
learning rate 1e-5, 10% linear warm-up, batch 2. Each word keeps its own box, as an OCR engine
would produce on a new document.

## Test results (seqeval entity level)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
{per_entity}
| **overall** | **{word["precision"]:.3f}** | **{word["recall"]:.3f}** | **{word["f1"]:.3f}** |

With segment-level boxes (every word takes its annotated entity box, the protocol behind the
published 90.3 F1) the same recipe reaches {segment["f1"]:.3f} F1; that setting relies on
ground-truth entity extents and is reported only for comparison.

## Usage

```python
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

repo = "AllanDai/freightlens-layoutlmv3-funsd"
processor = LayoutLMv3Processor.from_pretrained(repo, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(repo)
```

## License

The base model is CC BY-NC-SA 4.0, so these weights are for non-commercial use only.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--detector-config", default="configs/real_final.yaml")
    parser.add_argument("--user", default="AllanDai")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = load_config(args.detector_config)
    metrics = config.paths.artifacts / "metrics"
    detector_metrics = detector_config.paths.artifacts / "metrics"
    manifest = _read(detector_config.paths.data / "manifests/loco.json")

    build = Path("build/model-cards")
    detector_dir, layout_dir = build / "detector", build / "layoutlmv3"
    for directory in [detector_dir, layout_dir]:
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
    weights = detector_config.paths.models / "vision/pallet/weights"
    shutil.copyfile(weights / "best.pt", detector_dir / "freightlens-yolo11n-loco.pt")
    shutil.copyfile(weights / "best.onnx", detector_dir / "freightlens-yolo11n-loco.onnx")
    card = detector_card(
        detector_metrics, metrics, manifest, detector_config.vision.detector.epochs
    )
    (detector_dir / "README.md").write_text(card, encoding="utf-8")
    shutil.copytree(
        config.paths.models / "document/layoutlmv3-funsd",
        layout_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("training_history.json"),
    )
    (layout_dir / "README.md").write_text(layout_card(metrics), encoding="utf-8")
    if args.dry_run:
        print(f"Model cards written to {build}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    for name, folder in [
        ("freightlens-yolo11n-loco", detector_dir),
        ("freightlens-layoutlmv3-funsd", layout_dir),
    ]:
        repo = f"{args.user}/{name}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message="Publish weights")
        print(f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
