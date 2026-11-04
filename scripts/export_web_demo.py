"""Build the assets for the static browser demo in demo/web/.

- model/detector.onnx: the LOCO-trained YOLO11n exported to ONNX (640x640 input, no NMS);
  the page runs it client-side with onnxruntime-web.
- model/meta.json: class names and input size read from the exported model.
- examples/forms/*.jpg and forms.json: LayoutLMv3 (word boxes) predictions on FUNSD test
  forms, precomputed because the 500 MB model and OCR are too heavy for a browser.
- metrics.json: headline test metrics copied from the measured metric files. The browser
  model's own score comes from evaluating detector.onnx on the LOCO test split
  (metrics/pallet_detection_onnx.json).
"""

import argparse
import ast
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from freightlens.config import load_config
from freightlens.devices import resolve_device

WEB = Path("demo/web")
ENTITY_COLORS = {"QUESTION": "#2563eb", "ANSWER": "#16a34a", "HEADER": "#ea580c"}


def export_detector(config) -> None:
    from ultralytics import YOLO

    weights = config.paths.models / "vision/pallet/weights/best.pt"
    onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=640, opset=17, simplify=True))
    target = WEB / "model"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_path, target / "detector.onnx")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    meta = session.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(meta["names"])
    size = session.get_inputs()[0].shape[-1]
    payload = {"names": [names[i] for i in range(len(names))], "input_size": size}
    (target / "meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def export_forms(config, form_ids: list[str]) -> None:
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    from freightlens.document_ai.funsd_layoutlm import (
        FUNSD_LABELS,
        _encode,
        funsd_root,
        load_split,
    )

    samples = {s["id"]: s for s in load_split(funsd_root(config), "test")}
    checkpoint = config.paths.models / "document/layoutlmv3-funsd"
    device = resolve_device(config.document.funsd.device)
    processor = LayoutLMv3Processor.from_pretrained(checkpoint, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint).to(device).eval()
    output = WEB / "examples/forms"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for form_id in form_ids:
        sample = samples[form_id]
        batch = _encode(processor, [sample], device)
        with torch.no_grad():
            guesses = model(**batch).logits.argmax(-1)[0].cpu().tolist()
        predicted: dict[int, str] = {}
        for index, word_id in enumerate(batch.word_ids(0)):
            if word_id is not None and word_id not in predicted:
                predicted[word_id] = FUNSD_LABELS[guesses[index]]
        scored = sorted(predicted)
        correct = sum(predicted[i] == FUNSD_LABELS[sample["labels"][i]] for i in scored)

        image = Image.open(sample["image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        sx, sy = image.width / 1000, image.height / 1000
        entities: list[list[str]] = []
        previous = "O"
        for word_id in scored:
            label = predicted[word_id]
            if label == "O":
                previous = label
                continue
            kind = label[2:]
            x0, y0, x1, y1 = sample["boxes"][word_id]
            draw.rectangle(
                [x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=ENTITY_COLORS[kind], width=3
            )
            word = sample["words"][word_id]
            if label.startswith("B-") or previous == "O" or previous[2:] != kind:
                entities.append([kind.lower(), word])
            else:
                entities[-1][1] += " " + word
            previous = label
        image.thumbnail((900, 1200))
        image.save(output / f"{form_id}.jpg", quality=85)
        records.append(
            {
                "id": form_id,
                "image": f"examples/forms/{form_id}.jpg",
                "words": len(scored),
                "word_label_accuracy": round(correct / len(scored), 4),
                "entities": entities,
            }
        )
    (WEB / "forms.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def export_metrics(config, detector_config) -> None:
    """Copy headline test metrics for the page header from the measured metric files."""

    def read(cfg, name: str) -> dict:
        path = cfg.paths.artifacts / "metrics" / name
        return json.loads(path.read_text(encoding="utf-8"))

    payload = {
        "pallet_AP50": read(detector_config, "pallet_detection.json")["pallet_AP50"],
        "browser_pallet_AP50": read(detector_config, "pallet_detection_onnx.json")["pallet_AP50"],
        "funsd_f1": read(config, "document_layoutlm_funsd.json")["f1"],
        "count_threshold": read(detector_config, "pallet_count_calibration.json")["threshold"],
        "review_threshold": config.reconciliation.vision_threshold,
        "document_threshold": config.reconciliation.document_threshold,
    }
    (WEB / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# Synthetic shipping documents paired with real LOCO test photos for the "verify a
# shipment" tab. Each document's pallet count is the photo's annotated (ground-truth) count
# plus a chosen paperwork error, so the page can say whether a flag is real or a false alarm.
# Indices select clean layouts of three different document types.
SHIPMENTS = [
    {
        "index": 36,
        "photo": "loco_1.jpg",
        "source": "s1_1564563502.8986418.jpg",
        "paperwork_error": 0,
    },
    {"index": 25, "photo": "loco_2.jpg", "source": "s1_1564564588038,75.jpg", "paperwork_error": 3},
    {
        "index": 26,
        "photo": "loco_4.jpg",
        "source": "s4_1579165461.8499296.jpg",
        "paperwork_error": 0,
    },
]


def export_shipments(config) -> None:
    """Generate documents, extract fields with PaddleOCR + the rule parser, save results."""
    import tempfile

    from freightlens.document_ai.generator import generate_one
    from freightlens.io import read_json
    from freightlens.pipeline import analyze_document
    from freightlens.reconciliation.state import expected_from_document

    output = WEB / "examples/documents"
    output.mkdir(parents=True, exist_ok=True)
    labels = config.paths.data / "processed/loco/labels/test"
    records = []
    with tempfile.TemporaryDirectory() as scratch:
        for spec in SHIPMENTS:
            rows = (labels / f"{Path(spec['source']).stem}.txt").read_text().splitlines()
            truth = sum(1 for row in rows if row.split() and row.split()[0] == "2")
            annotation = generate_one(
                Path(scratch),
                spec["index"],
                config.seed,
                "demo",
                pallet_count=truth + spec["paperwork_error"],
            )
            printed = read_json(annotation)["fields"]
            document, _ = analyze_document(annotation.with_suffix(".png"), config)
            expected = expected_from_document(document)
            image = Image.open(annotation.with_suffix(".png")).convert("RGB")
            image.thumbnail((900, 1200))
            name = f"{document.shipment_id}.jpg"
            image.save(output / name, quality=85)
            extracted = document.model_dump(mode="json")
            records.append(
                {
                    "shipment_id": document.shipment_id,
                    "document_type": document.document_type,
                    "image": f"examples/documents/{name}",
                    "photo": f"examples/warehouse/{spec['photo']}",
                    "photo_source": f"LOCO test {spec['source']}",
                    "photo_annotated_pallets": truth,
                    "paperwork_error": spec["paperwork_error"],
                    "fields": [
                        {
                            "name": field,
                            "printed": printed.get(field),
                            "extracted": extracted.get(field),
                            "confidence": document.field_confidences.get(field),
                        }
                        for field in [
                            "shipment_id",
                            "carrier",
                            "vehicle_id",
                            "pallet_count",
                            "total_weight_lb",
                        ]
                    ],
                    "expected": expected.model_dump(mode="json"),
                }
            )
    (WEB / "shipments.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument(
        "--detector-config",
        default="configs/real_final.yaml",
        help="Config whose trained detector the page runs (default: the final model)",
    )
    parser.add_argument("--forms", nargs="+", default=["83594639", "86230203_0206", "82251504"])
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = load_config(args.detector_config)
    export_detector(detector_config)
    export_forms(config, args.forms)
    export_shipments(config)
    export_metrics(config, detector_config)


if __name__ == "__main__":
    main()
