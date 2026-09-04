import socket

import cv2

from freightlens.io import read_json
from freightlens.pipeline import run_demo
from freightlens.vision.tracker import track_video
from freightlens.vision.video_generator import generate_video


def test_cpu_pipeline_needs_no_network(isolated_config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline smoke attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    report = run_demo(isolated_config)
    assert report["observed"]["observed_pallet_count"] == 2
    assert report["verification"]["pallet_delta"] == -1
    assert report["verification"]["anomaly_type"] == "MISSING_PALLET"
    summary = read_json(isolated_config.paths.artifacts / "metrics/summary.json")
    assert summary["document_baseline"]["samples"] == 4
    assert summary["document_layoutlm"]["status"] == "not_run"
    capture = cv2.VideoCapture(str(isolated_config.paths.artifacts / "demo/tracking_demo.mp4"))
    assert capture.read()[0]
    capture.release()


def test_tracker_resets_between_videos(isolated_config):
    a = generate_video(isolated_config, 0)
    b = generate_video(isolated_config, 1)
    first = track_video(a, isolated_config)
    second = track_video(b, isolated_config)
    assert min(event["track_id"] for event in first["events"]) == 1
    assert min(event["track_id"] for event in second["events"]) == 1
