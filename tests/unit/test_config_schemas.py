import pytest
from pydantic import ValidationError

from freightlens.config import load_config
from freightlens.schemas.document import ShipmentDocument
from freightlens.schemas.vision import DetectionEvent


def test_smoke_inherits_and_overrides():
    config = load_config("configs/smoke.yaml")
    assert config.document.synthetic.train_size == 30
    assert config.vision.detector.model == "yolo11n.pt"
    assert config.reconciliation.document_threshold == 0.7


def test_config_cycle(tmp_path):
    path = tmp_path / "cycle.yaml"
    path.write_text("extends: cycle.yaml", encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)


@pytest.mark.parametrize("field,value", [("pallet_count", -1), ("extraction_confidence", 2)])
def test_document_rejects_invalid(field, value):
    with pytest.raises(ValidationError):
        ShipmentDocument(document_id="DOC", document_type="invoice", **{field: value})


def test_box_rejects_reversed_corners():
    with pytest.raises(ValidationError):
        DetectionEvent(
            video_id="V",
            frame_index=0,
            timestamp_sec=0,
            track_id=1,
            confidence=0.9,
            bbox_xyxy=(20, 10, 5, 15),
        )


def test_run_metadata_without_git(isolated_config, monkeypatch):
    from freightlens.io import read_json, record_run

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("freightlens.io.subprocess.run", unavailable)
    path = record_run(isolated_config, "no-git", {"status": "measured"})
    assert read_json(path / "environment.json")["git_commit"] is None
