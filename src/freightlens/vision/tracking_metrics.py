from collections import defaultdict

import numpy as np

from freightlens.config import Config
from freightlens.io import read_json, record_run, write_json
from freightlens.vision.metrics import match_boxes
from freightlens.vision.tracker import track_video


def evaluate_tracking(tracking: dict, truth: dict) -> dict:
    detections = defaultdict(list)
    for event in tracking["events"]:
        if event["event_type"] == "detected":
            detections[event["frame_index"]].append(event)
    assignments: dict[tuple[int, int], int] = {}
    previous: dict[int, int] = {}
    histories: dict[int, set[int]] = defaultdict(set)
    switches = 0
    for frame in truth["frames"]:
        predicted = detections[frame["frame_index"]]
        visible = [obj for obj in frame["objects"] if obj["visible"]]
        matches = match_boxes([p["bbox_xyxy"] for p in predicted], [t["bbox"] for t in visible])
        for p, t in matches:
            pred_id, true_id = predicted[p]["track_id"], visible[t]["track_id"]
            assignments[(frame["frame_index"], pred_id)] = true_id
            if true_id in previous and previous[true_id] != pred_id:
                switches += 1
            previous[true_id] = pred_id
            histories[true_id].add(pred_id)
    actual_events = [e for e in tracking["events"] if e["event_type"] == "crossed_line"]
    remaining = list(truth["crossings"])
    tp = 0
    for event in actual_events:
        true_id = assignments.get((event["frame_index"], event["track_id"]))
        match = next(
            (
                item
                for item in remaining
                if item["track_id"] == true_id
                and abs(item["frame_index"] - event["frame_index"]) <= 2
            ),
            None,
        )
        if match:
            remaining.remove(match)
            tp += 1
    precision = tp / len(actual_events) if actual_events else (1.0 if not truth["crossings"] else 0)
    recall = tp / len(truth["crossings"]) if truth["crossings"] else 1.0
    identities = {obj["track_id"] for frame in truth["frames"] for obj in frame["objects"]}
    fragmentation = sum(max(0, len(histories[key]) - 1) for key in identities) / max(
        len(identities), 1
    )
    return {
        "unique_count_error": abs(tracking["unique_count"] - truth["expected_unique_count"]),
        "crossing_precision": precision,
        "crossing_recall": recall,
        "crossing_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "id_switches": switches,
        "fragmentation_rate": fragmentation,
        "expected_count": truth["expected_unique_count"],
        "observed_count": tracking["unique_count"],
        "scenario": truth["scenario"],
        "video_id": truth["video_id"],
    }


def run_tracking(config: Config) -> dict:
    paths = [
        config.paths.data / "synthetic/tracking" / f"VIDEO_{i:03d}.mp4"
        for i in range(config.vision.video.count)
    ]
    if not paths or not all(path.exists() for path in paths):
        raise ValueError("Generate tracking videos first")
    rows, events = [], []
    for path in paths:
        truth = read_json(path.with_suffix(".json"))
        result = track_video(path, config, line=truth["line"], zone=truth["zone"])
        rows.append(evaluate_tracking(result, truth))
        events.extend(result["events"])
    output = config.paths.artifacts / "predictions/tracking_events.jsonl"
    import json

    output.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    numeric = [
        "unique_count_error",
        "crossing_precision",
        "crossing_recall",
        "crossing_f1",
        "fragmentation_rate",
    ]
    metrics = {
        "status": "measured",
        "backend": config.vision.tracker.type,
        "evaluation_scope": "fixture_pixel_tracking"
        if config.vision.tracker.type == "fixture"
        else "yolo_bytetrack_controlled_composites",
        "videos": len(rows),
        "per_video": rows,
        **{key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "id_switches": sum(row["id_switches"] for row in rows),
    }
    write_json(config.paths.artifacts / "metrics/tracking.json", metrics)
    record_run(config, "tracking", metrics)
    return metrics
