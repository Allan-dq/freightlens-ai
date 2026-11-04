"""FreightLens AI demo: real-warehouse detection and scanned-form understanding.

Tab 1 runs the YOLO11n detector fine-tuned on LOCO (real logistics scenes).
Tab 2 runs LayoutLMv3 fine-tuned on FUNSD (real scanned forms) on top of Tesseract OCR.
"""

import os
import time
from collections import Counter
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
DETECTOR_PATH = os.environ.get("FREIGHTLENS_DETECTOR", str(HERE / "models/loco_yolo11n.pt"))
LAYOUT_MODEL = os.environ.get("FREIGHTLENS_LAYOUT_MODEL", str(HERE / "models/layoutlmv3-funsd"))
COLORS = {
    "pallet": "#22c55e",
    "small_load_carrier": "#eab308",
    "stillage": "#3b82f6",
    "forklift": "#f97316",
    "pallet_truck": "#06b6d4",
}
ENTITY_COLORS = {"QUESTION": "#3b82f6", "ANSWER": "#22c55e", "HEADER": "#f97316"}

_detector = None
_layout = None


def detector():
    global _detector
    if _detector is None:
        from ultralytics import YOLO

        _detector = YOLO(DETECTOR_PATH)
    return _detector


def layout():
    global _layout
    if _layout is None:
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3ImageProcessor,
            LayoutLMv3Processor,
        )

        ocr = LayoutLMv3ImageProcessor(apply_ocr=True)
        processor = LayoutLMv3Processor.from_pretrained(LAYOUT_MODEL, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(LAYOUT_MODEL).eval()
        _layout = (ocr, processor, model)
    return _layout


def detect(image: Image.Image, confidence: float):
    if image is None:
        return None, "Upload an image first.", None
    model = detector()
    start = time.perf_counter()
    result = model.predict(image, conf=confidence, imgsz=640, device="cpu", verbose=False)[0]
    elapsed = (time.perf_counter() - start) * 1000
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, canvas.width // 400)
    counts: Counter = Counter()
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        counts[name] += 1
        draw.rectangle(box.xyxy[0].tolist(), outline=COLORS.get(name, "#ffffff"), width=width)
    rows = [[name, counts[name]] for name in COLORS if counts[name]]
    summary = f"{sum(counts.values())} objects in {elapsed:.0f} ms on CPU"
    return canvas, summary, rows


def understand(image: Image.Image):
    if image is None:
        return None, "Upload a form image first.", None
    import torch

    ocr, processor, model = layout()
    image = image.convert("RGB")
    start = time.perf_counter()
    features = ocr(image)
    words, boxes = features["words"][0], features["boxes"][0]
    if not words:
        return image, "No text found by OCR.", None
    encoding = processor(
        image, words, boxes=boxes, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        logits = model(**encoding).logits[0]
    elapsed = (time.perf_counter() - start) * 1000
    predictions = logits.argmax(-1).tolist()
    word_labels: dict[int, str] = {}
    for index, word_id in enumerate(encoding.word_ids(0)):
        if word_id is not None and word_id not in word_labels:
            word_labels[word_id] = model.config.id2label[predictions[index]]
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    scale_x, scale_y = image.width / 1000, image.height / 1000
    entities: list[list[str]] = []
    previous = "O"
    for word_id in sorted(word_labels):
        label = word_labels[word_id]
        if label == "O":
            previous = label
            continue
        kind = label.split("-", 1)[1]
        x0, y0, x1, y1 = boxes[word_id]
        draw.rectangle(
            [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
            outline=ENTITY_COLORS[kind],
            width=2,
        )
        if label.startswith("B-") or previous == "O" or previous.split("-", 1)[1] != kind:
            entities.append([kind.lower(), words[word_id]])
        else:
            entities[-1][1] += " " + words[word_id]
        previous = label
    scored = len(word_labels)
    note = (
        f" ({len(words) - scored} words beyond 512 tokens skipped)" if scored < len(words) else ""
    )
    summary = (
        f"{len(entities)} entities from {len(words)} OCR words in {elapsed:.0f} ms on CPU{note}"
    )
    return canvas, summary, entities


EXAMPLES = HERE / "examples"
warehouse_examples = sorted(str(p) for p in (EXAMPLES / "warehouse").glob("*.jpg"))
form_examples = sorted(str(p) for p in (EXAMPLES / "forms").glob("*.png"))

with gr.Blocks(title="FreightLens AI") as demo:
    gr.Markdown(
        "# FreightLens AI\n"
        "Models fine-tuned on **real public data**: a YOLO11n detector trained on the LOCO "
        "warehouse dataset and a LayoutLMv3 form model trained on FUNSD scanned forms. "
        "Metrics and training code: [github.com/Allan-dq/freightlens-ai]"
        "(https://github.com/Allan-dq/freightlens-ai)."
    )
    with gr.Tab("Warehouse objects"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Warehouse photo")
                confidence = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence")
                run_detect = gr.Button("Detect", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Detections")
                detect_summary = gr.Markdown()
                detect_table = gr.Dataframe(headers=["class", "count"], label="Counts")
        if warehouse_examples:
            gr.Examples(warehouse_examples, inputs=image_in)
        run_detect.click(detect, [image_in, confidence], [image_out, detect_summary, detect_table])
    with gr.Tab("Form understanding"):
        with gr.Row():
            with gr.Column():
                form_in = gr.Image(type="pil", label="Scanned form")
                run_form = gr.Button("Extract", variant="primary")
            with gr.Column():
                form_out = gr.Image(label="Entities (blue question, green answer, orange header)")
                form_summary = gr.Markdown()
                form_table = gr.Dataframe(headers=["entity", "text"], label="Entities")
        if form_examples:
            gr.Examples(form_examples, inputs=form_in)
        run_form.click(understand, [form_in], [form_out, form_summary, form_table])

if __name__ == "__main__":
    demo.launch()
