"""Opt-in pretrained model integration. Never invoked by offline CI."""

import shutil

from freightlens.config import load_config
from freightlens.document_ai.layoutlm import train_layoutlm
from freightlens.document_ai.metrics import evaluate_documents
from freightlens.io import write_json
from freightlens.vision.detector import train_detector
from freightlens.vision.palleset import download_subset, prepare_palleset
from freightlens.vision.tracker import track_video

if __name__ == "__main__":
    config = load_config("configs/smoke.yaml")
    from freightlens.pipeline import run_demo

    run_demo(config)
    download_subset(config)
    prepare_palleset(config)
    detector = train_detector(config)
    layout = train_layoutlm(config, force=True)
    config.vision.tracker.type = "bytetrack"
    tracking = track_video(
        config.paths.data / "synthetic/tracking/VIDEO_000.mp4",
        config,
        output=config.paths.artifacts / "videos/bytetrack_model_probe.mp4",
    )
    config.document.ocr.backend = "paddleocr"
    config.paths.artifacts = config.paths.artifacts / "model-probe"
    ocr = evaluate_documents(config)
    shutil.copyfile(
        config.paths.artifacts / "metrics/document_baseline.json",
        config.paths.artifacts.parent / "metrics/document_paddleocr.json",
    )
    write_json(
        config.paths.artifacts / "integration.json",
        {
            "detector": detector,
            "layoutlm": layout,
            "ocr": ocr,
            "bytetrack": {
                "frames": tracking["frames"],
                "events": len(tracking["events"]),
                "note": "Few-step detector can yield no detections.",
            },
        },
    )
