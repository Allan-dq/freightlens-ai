import numpy as np
import pytest

from freightlens.schemas.vision import DetectionEvent
from freightlens.vision.counting import CrossingCounter
from freightlens.vision.geometry import crosses_segment, point_in_polygon
from freightlens.vision.palleset import identify_pairs, read_boxes, select_splits, to_yolo
from freightlens.vision.tracking_metrics import evaluate_tracking


def event(x, frame, track=1, confidence=0.9):
    return DetectionEvent(
        video_id="v",
        frame_index=frame,
        timestamp_sec=frame / 8,
        track_id=track,
        confidence=confidence,
        bbox_xyxy=(x - 2, 10, x + 2, 20),
    )


def test_count_direction_and_line_dwell_and_duplicate():
    counter = CrossingCounter([[50, 0], [50, 100]], [[50, 0], [90, 0], [90, 90], [50, 90]])
    events = []
    for frame, x in enumerate([20, 50, 60, 30, 70]):
        events.extend(counter.update(event(x, frame)))
    assert sum(item.event_type == "crossed_line" for item in events) == 1
    assert len(counter.counted) == 1
    reverse = CrossingCounter([[50, 0], [50, 100]], [[0, 0], [90, 0], [90, 90]])
    reverse.update(event(80, 0))
    reverse.update(event(20, 1))
    assert not reverse.counted


def test_low_confidence_ignored_and_finite_segment():
    counter = CrossingCounter([[50, 0], [50, 100]], [[0, 0], [90, 0], [90, 90]])
    assert not counter.update(event(20, 0, confidence=0.1))
    assert not crosses_segment((40, 200), (60, 200), [[50, 0], [50, 100]])
    assert point_in_polygon((5, 5), [(0, 0), (10, 0), (10, 10), (0, 10)])
    assert not point_in_polygon((20, 5), [(0, 0), (10, 0), (10, 10)])


def test_pairing_split_stability_and_no_leakage():
    names = [
        f"{split}/{directory}/{prefix}_{i:04d}.{suffix}"
        for split in ["train", "val"]
        for i in range(8)
        for directory, prefix, suffix in [
            ("rgb", "rgb", "png"),
            ("bboxes_2d", "bounding_box_2d_tight", "npy"),
        ]
    ]
    pairs = identify_pairs(names)
    a = select_splits(pairs, 3, 2, 2, 42)
    b = select_splits(identify_pairs(names[::-1]), 3, 2, 2, 42)
    assert a == b
    all_ids = sum(a.values(), [])
    assert len(set(all_ids)) == len(all_ids)


def test_npy_conversion_and_bbox_validation(tmp_path):
    box = np.array(
        [(0, -5, 10, 90, 50, 0.1)],
        dtype=[
            ("semanticId", "u4"),
            ("x_min", "i4"),
            ("y_min", "i4"),
            ("x_max", "i4"),
            ("y_max", "i4"),
            ("occlusionRatio", "f4"),
        ],
    )
    path = tmp_path / "boxes.npy"
    np.save(path, box)
    assert read_boxes(path, 100, 100) == [[0, 10, 90, 50]]
    assert to_yolo([0, 0, 100, 100], 100, 100) == [0.5, 0.5, 1, 1]
    with pytest.raises(ValueError):
        to_yolo([0, 0, 120, 100], 100, 100)


def test_metric_detects_id_switch_and_fragmentation():
    truth = {
        "frames": [
            {
                "frame_index": f,
                "objects": [{"track_id": 1, "bbox": [10, 10, 30, 30], "visible": True}],
            }
            for f in range(2)
        ],
        "crossings": [],
        "expected_unique_count": 0,
        "scenario": "unit",
        "video_id": "v",
    }
    predicted = {
        "unique_count": 0,
        "events": [
            {
                "frame_index": f,
                "track_id": f + 1,
                "bbox_xyxy": [10, 10, 30, 30],
                "event_type": "detected",
            }
            for f in range(2)
        ],
    }
    metric = evaluate_tracking(predicted, truth)
    assert metric["id_switches"] == 1
    assert metric["fragmentation_rate"] == 1
