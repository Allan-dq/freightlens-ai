"""Separate qualitative OCR validation on real scanned FUNSD forms."""

from freightlens.config import Config
from freightlens.document_ai.metrics import character_error_rate
from freightlens.document_ai.ocr import PaddleOCRBackend
from freightlens.document_ai.preprocess import load_pages
from freightlens.io import read_json, write_json


def evaluate_funsd(config: Config, limit: int = 2) -> dict:
    root = config.paths.data / "raw/funsd"
    annotations = sorted(
        path
        for path in root.glob("**/testing_data/annotations/*.json")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )[:limit]
    if not annotations:
        raise ValueError("Download FUNSD before external validation")
    engine = PaddleOCRBackend()
    examples = []
    for path in annotations:
        image = path.parent.parent / "images" / f"{path.stem}.png"
        truth = read_json(path)
        expected = " ".join(word["text"] for field in truth["form"] for word in field["words"])
        prediction = engine.extract(load_pages(image)[0])
        examples.append(
            {
                "id": path.stem,
                "ground_truth_text": expected,
                "ocr": prediction.model_dump(),
                "cer_annotation_order": character_error_rate(expected, prediction.text),
            }
        )
    result = {
        "status": "measured",
        "samples": len(examples),
        "source": "FUNSD real scanned forms",
        "evaluation_scope": "external_ocr_qualitative_not_freight",
        "note": "CER also reflects differences between annotation order and OCR reading order.",
        "examples": examples,
    }
    write_json(config.paths.artifacts / "metrics/funsd_ocr.json", result)
    return result
