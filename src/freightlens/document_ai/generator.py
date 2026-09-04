"""Original vector freight forms with pixel-aligned field and token labels."""

import random
from pathlib import Path

import numpy as np
from faker import Faker
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from freightlens.config import Config
from freightlens.document_ai.degradation import degrade, transform_box
from freightlens.document_ai.preprocess import load_pages
from freightlens.io import read_json, sha256, write_json

TYPES = ["bill_of_lading", "advance_shipping_notice", "invoice", "packing_list"]
FIELDS = {
    "shipment_id": "Shipment ID",
    "carrier": "Carrier",
    "vehicle_id": "Vehicle ID",
    "origin": "Origin",
    "destination": "Destination",
    "pallet_count": "Pallets",
    "package_count": "Packages",
    "total_weight_lb": "Total Weight",
}
PAGE = (612, 792)


def generate_one(
    directory: Path, index: int, seed: int, split: str, pallet_count: int | None = None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + index)
    fake = Faker("en_US")
    fake.seed_instance(seed + index)
    doc_id = f"DOC_{index:06d}"
    kind = TYPES[index % len(TYPES)]
    layout = (index // len(TYPES)) % 3
    level = ["clean", "light", "medium", "hard"][(index // 3) % 4]
    fields = {
        "shipment_id": f"SHP-{index:06d}",
        "carrier": f"{fake.last_name()} Freight",
        "vehicle_id": f"TR-{rng.randint(1000, 9999)}",
        "origin": f"{fake.city()}, {fake.state_abbr()}",
        "destination": f"{fake.city()}, {fake.state_abbr()}",
        "pallet_count": pallet_count if pallet_count is not None else rng.randint(1, 24),
        "package_count": rng.randint(20, 280),
        "total_weight_lb": float(rng.randint(300, 9500)),
    }
    if index % 7 == 0:
        fields["vehicle_id"] = None
    pdf = directory / f"{doc_id}.pdf"
    canvas = Canvas(str(pdf), pagesize=PAGE, invariant=1)
    canvas.setTitle(f"FreightLens synthetic {kind}")
    tokens = []
    field_boxes = {}
    font = ["Helvetica", "Courier", "Times-Roman"][layout]
    font_size = [10, 10, 11][layout]

    def text(x, y, value, label="O", size=font_size, selected_font=font):
        canvas.setFont(selected_font, size)
        canvas.drawString(x, y, value)
        cursor = x
        for word_index, word in enumerate(value.split()):
            width = stringWidth(word, selected_font, size)
            entity = label if label == "O" else f"{'B' if word_index == 0 else 'I'}-{label}"
            tokens.append(
                {
                    "text": word,
                    "box": [
                        cursor * 1.5,
                        (792 - y - size) * 1.5,
                        (cursor + width) * 1.5,
                        (792 - y + 3) * 1.5,
                    ],
                    "label": entity,
                }
            )
            cursor += width + stringWidth(" ", selected_font, size)

    canvas.setFillColor(HexColor("#102E40"))
    canvas.rect(0, 674, 612, 118, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#6CE0C4"))
    text(40 if layout != 2 else 305, 752, "F / L", size=16, selected_font="Helvetica-Bold")
    canvas.setFillColor(HexColor("#FFFFFF"))
    text(40, 715, kind.replace("_", " ").upper(), size=19, selected_font="Helvetica-Bold")
    text(40, 693, f"{doc_id}  /  ORIGINAL SYNTHETIC FORM", size=9)
    canvas.setFillColor(HexColor("#253C49"))
    text(
        40, 648, f"Document number: FL-{index:07d}     Purchase order: PO-{100000 + index}", size=9
    )
    text(40, 627, f"Ship date: 2026-01-{1 + index % 25:02d}     Arrival date: 2026-02-01", size=9)
    items = list(FIELDS.items())
    if layout == 1:
        items = items[3:5] + items[:3] + items[5:]
    elif layout == 2:
        items = items[:1] + items[5:] + items[1:5]
    y = 590
    for field, label in items:
        value = fields[field]
        if value is None:
            continue
        canvas.setFillColor(HexColor("#667E8B"))
        text(40, y, label + ":")
        canvas.setFillColor(HexColor("#102E40"))
        formatted = f"{value:,.0f} lb" if field == "total_weight_lb" else str(value)
        x = 183 if layout != 2 else 210
        text(x, y, formatted, field.upper())
        field_boxes[field] = [
            x * 1.5,
            (792 - y - font_size) * 1.5,
            (x + stringWidth(formatted, font, font_size)) * 1.5,
            (792 - y + 3) * 1.5,
        ]
        y -= [27, 29, 25][layout]
    rows = rng.randint(3, 6)
    top = min(y - 15, 340)
    canvas.setFillColor(HexColor("#E6F3F0"))
    canvas.rect(40, top - 7, 532, 25, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#102E40"))
    headers = [(48, "SKU"), (145, "Description"), (370, "Qty"), (430, "Unit USD"), (509, "Total")]
    for x, label in headers:
        text(x, top, label, size=9)
    line_items = []
    for row in range(rows):
        y_row = top - (row + 1) * 23
        quantity, price = rng.randint(1, 20), rng.randint(8, 90)
        description = rng.choice(["Packed components", "Cartoned goods", "Wrapped equipment"])
        item = [
            f"SKU-{index:04d}-{row}",
            description,
            str(quantity),
            f"{price:.2f}",
            f"{quantity * price:.2f}",
        ]
        for (x, _), value in zip(headers, item, strict=True):
            text(x, y_row, value, size=9)
        canvas.setStrokeColor(HexColor("#DAE4E8"))
        canvas.line(40, y_row - 7, 572, y_row - 7)
        line_items.append(
            dict(
                zip(
                    ["sku", "description", "quantity", "unit_price", "line_total"],
                    item,
                    strict=True,
                )
            )
        )
    canvas.setFillColor(HexColor("#667E8B"))
    text(40, 60, "Generated research sample. No real company or shipment is represented.", size=8)
    text(40, 43, "FreightLens AI  /  Expected shipment record", size=8)
    page_count = 2 if index % 9 == 0 else 1
    text(516, 43, f"1 / {page_count}", size=8)
    if page_count == 2:
        canvas.showPage()
        canvas.setFillColor(HexColor("#102E40"))
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawString(40, 732, "HANDLING ADDENDUM")
        canvas.setFont("Helvetica", 11)
        canvas.drawString(40, 695, f"Reference {doc_id}. Keep packages dry and upright.")
        canvas.drawString(40, 670, "Synthetic continuation page; shipment fields appear on page 1.")
        canvas.drawString(520, 43, "2 / 2")
    canvas.save()
    clean = load_pages(pdf)[0]
    noisy, degradation, matrix = degrade(clean, level, seed + index)
    image_path = directory / f"{doc_id}.png"
    Image.fromarray(noisy).save(image_path)
    clean_path = directory / f"{doc_id}_clean.png"
    Image.fromarray(clean).save(clean_path)
    height, width = noisy.shape[:2]
    for token in tokens:
        token["clean_box"] = token["box"]
        token["box"] = transform_box(token["box"], matrix, width, height)
    return write_json(
        directory / f"{doc_id}.json",
        {
            "id": doc_id,
            "document_type": kind,
            "fields": fields,
            "split": split,
            "seed": seed + index,
            "layout": layout,
            "degradation": degradation.model_dump(),
            "clean_id": f"{doc_id}_clean",
            "noisy_id": doc_id,
            "page_count": page_count,
            "image": image_path.name,
            "pdf": pdf.name,
            "image_size": [width, height],
            "tokens": tokens,
            "field_boxes": field_boxes,
            "line_items": line_items,
            "clean_text": " ".join(token["text"] for token in tokens),
            "homography": np.round(matrix, 8).tolist(),
        },
    )


def generate_documents(config: Config) -> list[Path]:
    manifest_path = config.paths.data / "manifests/documents.json"
    requested = config.document.synthetic.model_dump()
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        previous_sizes = manifest.get("sizes")
        if manifest["seed"] != config.seed or (previous_sizes and previous_sizes != requested):
            raise ValueError(
                "Document seed or split changed. Use a new data root to preserve "
                "existing evaluation splits."
            )
    paths = []
    index = 1
    for split in ["train", "val", "test"]:
        size = getattr(config.document.synthetic, f"{split}_size")
        for _ in range(size):
            paths.append(
                generate_one(
                    config.paths.data / "synthetic/documents" / split, index, config.seed, split
                )
            )
            index += 1
    write_json(
        manifest_path,
        {
            "seed": config.seed,
            "sizes": requested,
            "samples": [{"path": path.as_posix(), "sha256": sha256(path)} for path in paths],
        },
    )
    return paths
