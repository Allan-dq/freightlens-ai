import re
from pathlib import Path

from freightlens.document_ai.generator import FIELDS, TYPES
from freightlens.document_ai.normalize import parse_number
from freightlens.document_ai.ocr import OCRResult, OCRToken
from freightlens.schemas.document import ShipmentDocument

ALIASES = {
    **{key: [value.lower()] for key, value in FIELDS.items()},
    "pallet_count": ["pallets", "pallet count", "total pallets"],
    "package_count": ["packages", "package count"],
    "total_weight_lb": ["total weight", "weight"],
}


def group_lines(result: OCRResult):
    lines: list[list[OCRToken]] = []
    for token in sorted(result.tokens, key=lambda t: (t.page, t.box[1], t.box[0])):
        cy = (token.box[1] + token.box[3]) / 2
        target = next(
            (
                line
                for line in lines
                if line[0].page == token.page
                and abs((line[0].box[1] + line[0].box[3]) / 2 - cy)
                < max(6, (token.box[3] - token.box[1]) * 0.7)
            ),
            None,
        )
        if target is None:
            lines.append([token])
        else:
            target.append(token)
    return [sorted(line, key=lambda t: t.box[0]) for line in lines]


def parse_document(result: OCRResult, document_id: str = "uploaded") -> ShipmentDocument:
    lines = group_lines(result)
    full_text = " ".join(t.text for line in lines for t in line).lower()
    kind = next((kind for kind in TYPES if kind.replace("_", " ") in full_text), None)
    values = {}
    confidences: dict[str, float] = {}
    value: str | int | float
    for field, aliases in ALIASES.items():
        for line_index, line in enumerate(lines):
            line_text = " ".join(t.text for t in line)
            label_match = next(
                (
                    re.match(r"^" + re.escape(label) + r"\b\s*:?\s*(.*)$", line_text, re.IGNORECASE)
                    for label in aliases
                    if re.match(r"^" + re.escape(label) + r"\b", line_text, re.IGNORECASE)
                ),
                None,
            )
            if label_match is None:
                continue
            raw = label_match[1].strip()
            spatial = 1.0
            value_tokens = line
            if not raw and line_index + 1 < len(lines):
                nearby = lines[line_index + 1]
                gap = nearby[0].box[1] - line[0].box[3]
                if 0 <= gap < 45 and abs(nearby[0].box[0] - line[0].box[0]) < 100:
                    raw = " ".join(t.text for t in nearby)
                    value_tokens = nearby
                    spatial = 0.8
            if not raw:
                continue
            try:
                if field in ["pallet_count", "package_count", "total_weight_lb"]:
                    value = parse_number(raw, integer=field != "total_weight_lb")
                else:
                    value = raw
                    if field in ["shipment_id", "vehicle_id"] and not re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9 -]{1,40}", raw
                    ):
                        raise ValueError("Unrecognized identifier")
                confidence = min(t.confidence for t in value_tokens) * spatial
                if confidence > confidences.get(field, -1):
                    values[field] = value
                    confidences[field] = confidence
            except ValueError:
                confidences.setdefault(field, 0.0)
    # Unknown document type receives zero confidence rather than a confident invented type.
    confidence = sum(confidences.get(field, 0) for field in FIELDS) / len(FIELDS)
    return ShipmentDocument.model_validate(
        {
            "document_id": Path(document_id).stem,
            "document_type": kind or "bill_of_lading",
            **values,
            "extraction_confidence": confidence if kind else 0,
            "field_confidences": {field: confidences.get(field, 0) for field in FIELDS},
        }
    )
