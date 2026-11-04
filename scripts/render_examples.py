"""Render README example images from the trained models on held-out test data.

- docs/images/loco_detections.jpg: detector output on a LOCO test-session image.
- docs/images/funsd_entities.jpg: LayoutLMv3 entity predictions on a FUNSD test form.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from freightlens.config import load_config
from freightlens.devices import resolve_device

CLASS_COLORS = {
    "pallet": "#22c55e",
    "small_load_carrier": "#eab308",
    "stillage": "#3b82f6",
    "forklift": "#f97316",
    "pallet_truck": "#06b6d4",
}
ENTITY_COLORS = {"QUESTION": "#3b82f6", "ANSWER": "#22c55e", "HEADER": "#f97316"}


def render_detections(config, image_name: str, output: Path) -> None:
    from freightlens.vision.detector import load_detector, trained_checkpoint

    model = load_detector(trained_checkpoint(config))
    path = config.paths.data / "processed/loco/images/test" / image_name
    result = model.predict(
        str(path),
        imgsz=config.vision.detector.image_size,
        conf=config.vision.tracker.confidence_threshold,
        device=resolve_device(config.vision.detector.device),
        verbose=False,
    )[0]
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        draw.rectangle(box.xyxy[0].tolist(), outline=CLASS_COLORS.get(name, "white"), width=4)
    image.thumbnail((1280, 1280))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=88)


def render_form(config, form_id: str, output: Path) -> None:
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    from freightlens.document_ai.funsd_layoutlm import (
        FUNSD_LABELS,
        _encode,
        funsd_root,
        load_split,
    )

    sample = next(s for s in load_split(funsd_root(config), "test") if s["id"] == form_id)
    checkpoint = config.paths.models / "document/layoutlmv3-funsd"
    device = resolve_device(config.document.funsd.device)
    processor = LayoutLMv3Processor.from_pretrained(checkpoint, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint).to(device).eval()
    batch = _encode(processor, [sample], device)
    with torch.no_grad():
        guesses = model(**batch).logits.argmax(-1)[0].cpu().tolist()
    labels: dict[int, str] = {}
    for index, word_id in enumerate(batch.word_ids(0)):
        if word_id is not None and word_id not in labels:
            labels[word_id] = FUNSD_LABELS[guesses[index]]
    image = Image.open(sample["image"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    sx, sy = image.width / 1000, image.height / 1000
    for word_id, label in labels.items():
        if label == "O":
            continue
        x0, y0, x1, y1 = sample["boxes"][word_id]
        draw.rectangle(
            [x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=ENTITY_COLORS[label[2:]], width=3
        )
    image.thumbnail((1000, 1000))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--loco-image", required=True, help="File name in the LOCO test split")
    parser.add_argument("--funsd-form", required=True, help="FUNSD test form id")
    args = parser.parse_args()
    config = load_config(args.config)
    render_detections(config, args.loco_image, Path("docs/images/loco_detections.jpg"))
    render_form(config, args.funsd_form, Path("docs/images/funsd_entities.jpg"))


if __name__ == "__main__":
    main()
