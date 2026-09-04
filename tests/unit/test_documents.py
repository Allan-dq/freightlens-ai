from pathlib import Path

import numpy as np
import pytest

from freightlens.document_ai.degradation import degrade, transform_box
from freightlens.document_ai.generator import generate_one
from freightlens.document_ai.layoutlm import prepare_sample
from freightlens.document_ai.metrics import character_error_rate
from freightlens.document_ai.normalize import parse_number
from freightlens.document_ai.ocr import FixtureOCRBackend, OCRResult, OCRToken
from freightlens.document_ai.parser import parse_document
from freightlens.document_ai.preprocess import load_pages
from freightlens.io import read_json, sha256


@pytest.mark.parametrize(
    "value,integer,expected",
    [("3,240 lbs", False, 3240.0), ("08 pallets", True, 8), ("10 kg", False, 22.05)],
)
def test_numeric_normalization(value, integer, expected):
    assert parse_number(value, integer) == expected


@pytest.mark.parametrize("value", ["12.5 pallets", "-4", "8 bananas", "2 3"])
def test_ambiguous_count_rejected(value):
    with pytest.raises(ValueError):
        parse_number(value, integer=True)


def test_generator_reproducible_and_parser_uses_tokens(tmp_path):
    path = generate_one(tmp_path, 1, 42, "test")
    first = sha256(path)
    image = load_pages(path.with_suffix(".png"))[0]
    result = FixtureOCRBackend(path, corrupt=False).extract(image)
    parsed = parse_document(result, path.stem)
    truth = read_json(path)
    assert parsed.shipment_id == truth["fields"]["shipment_id"]
    assert parsed.pallet_count == truth["fields"]["pallet_count"]
    generate_one(tmp_path, 1, 42, "test")
    assert sha256(path) == first
    sample = prepare_sample(path)
    assert len(sample["words"]) == len(sample["labels"]) == len(sample["boxes"])
    assert all(0 <= value <= 1000 for box in sample["boxes"] for value in box)


def test_spatial_below_label_and_missing_fields():
    tokens = [
        OCRToken(text="PACKING LIST", box=(0, 0, 150, 20), confidence=1),
        OCRToken(text="Pallets:", box=(0, 40, 70, 60), confidence=0.9),
        OCRToken(text="8", box=(0, 70, 20, 90), confidence=0.8),
    ]
    result = parse_document(OCRResult(tokens=tokens, backend="test"))
    assert result.pallet_count == 8
    assert result.shipment_id is None
    assert result.field_confidences["pallet_count"] == pytest.approx(0.64)
    assert result.extraction_confidence < 0.7


def test_degradation_reproducibility_and_transform():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    a, config_a, matrix = degrade(image, "hard", 3)
    b, config_b, _ = degrade(image, "hard", 3)
    assert np.array_equal(a, b)
    assert config_a == config_b
    box = transform_box([10, 10, 80, 80], matrix, 100, 100)
    assert 0 <= box[0] < box[2] <= 100
    assert not np.array_equal(image, a)


def test_character_error_rate():
    assert character_error_rate("pallet", "palet") == pytest.approx(1 / 6)
    assert character_error_rate("", "") == 0


def test_arbitrary_image_never_uses_ground_truth(tmp_path):
    from freightlens.config import load_config
    from freightlens.pipeline import analyze_document

    with pytest.raises(ValueError, match="arbitrary"):
        analyze_document(Path("arbitrary.png"), load_config("configs/smoke.yaml"))
