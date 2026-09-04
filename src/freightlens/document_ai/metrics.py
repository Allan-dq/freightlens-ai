from collections import defaultdict

import numpy as np

from freightlens.config import Config
from freightlens.document_ai.generator import FIELDS
from freightlens.document_ai.normalize import normalize_text
from freightlens.document_ai.ocr import FixtureOCRBackend, PaddleOCRBackend
from freightlens.document_ai.parser import parse_document
from freightlens.document_ai.preprocess import load_pages
from freightlens.io import read_json, record_run, write_json


def character_error_rate(expected: str, actual: str) -> float:
    previous = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        current = [i]
        for j, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(len(expected), 1)


def field_metrics(rows):
    tp = sum(row["normalized"] for row in rows if row["expected"] is not None)
    predicted = sum(row["actual"] is not None for row in rows)
    positive = sum(row["expected"] is not None for row in rows)
    precision, recall = tp / max(predicted, 1), tp / max(positive, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exact_match": sum(r["exact"] for r in rows) / max(len(rows), 1),
        "normalized_exact_match": sum(r["normalized"] for r in rows) / max(len(rows), 1),
        "fields": len(rows),
    }


def evaluate_documents(config: Config, backend: str = "rules") -> dict:
    if backend == "layoutlm":
        from freightlens.document_ai.layoutlm import evaluate_layoutlm

        return evaluate_layoutlm(config)
    paths = sorted((config.paths.data / "synthetic/documents/test").glob("DOC_*.json"))
    if not paths:
        raise ValueError("Generate documents before evaluation")
    paddle = PaddleOCRBackend() if config.document.ocr.backend == "paddleocr" else None
    rows, errors, predictions, cers, all_correct = [], [], [], [], []
    groups = defaultdict(list)
    for path in paths:
        truth = read_json(path)
        engine = paddle or FixtureOCRBackend(path)
        ocr = engine.extract(load_pages(path.with_suffix(".png"))[0])
        predicted = parse_document(ocr, truth["id"])
        predictions.append(predicted.model_dump())
        document_rows = []
        for field in FIELDS:
            expected, actual = truth["fields"].get(field), getattr(predicted, field)
            row = {
                "expected": expected,
                "actual": actual,
                "exact": expected == actual,
                "normalized": normalize_text(expected) == normalize_text(actual),
            }
            document_rows.append(row)
            groups[f"field/{field}"].append(row)
            if not row["normalized"]:
                errors.append(
                    {
                        "id": truth["id"],
                        "field": field,
                        **row,
                        "degradation": truth["degradation"]["level"],
                        "ocr_text": ocr.text,
                        "source": str(path),
                    }
                )
        rows.extend(document_rows)
        groups[f"type/{truth['document_type']}"].extend(document_rows)
        groups[f"degradation/{truth['degradation']['level']}"].extend(document_rows)
        all_correct.append(all(row["normalized"] for row in document_rows))
        cers.append(character_error_rate(truth["clean_text"], ocr.text))
    metrics = {
        "status": "measured",
        "backend": config.document.ocr.backend,
        "evaluation_scope": "fixture_parser_only" if paddle is None else "raster_ocr_and_parser",
        "samples": len(paths),
        **field_metrics(rows),
        "document_all_correct": float(np.mean(all_correct)),
        "ocr_cer": float(np.mean(cers)),
        "breakdown": {key: field_metrics(value) for key, value in groups.items()},
    }
    write_json(config.paths.artifacts / "metrics/document_baseline.json", metrics)
    write_json(config.paths.artifacts / "predictions/document_errors.json", errors)
    write_json(config.paths.artifacts / "predictions/documents.json", predictions)
    record_run(config, "document-baseline", metrics)
    return metrics
