"""Controlled trajectories, optionally composited from public Palleset crops."""

import math
from pathlib import Path

import cv2
import numpy as np

from freightlens.config import Config
from freightlens.io import read_json, write_json
from freightlens.vision.geometry import crosses_segment, side

SCENARIOS = [
    "normal_crossing",
    "two_pallets_close",
    "temporary_occlusion",
    "partial_exit",
    "duplicate_visual_overlap",
    "reverse_direction",
    "crowded_crossing",
]


def _public_crop(config: Config):
    from freightlens.vision.palleset import read_boxes

    manifest_path = config.paths.data / "manifests/palleset_subset.json"
    if not manifest_path.exists():
        raise ValueError("Full video generation requires downloaded Palleset crops")
    manifest = read_json(manifest_path)
    # Only training imagery is used for composites, never held-out detector images.
    root = config.paths.data / "raw/palleset"
    for key in manifest["train_ids"]:
        pair = manifest["pairs"][key]
        image = cv2.imread(str(root / pair["rgb"]))
        if image is None:
            continue
        h, w = image.shape[:2]
        boxes = read_boxes(root / pair["bboxes_2d"], w, h)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if x2 - x1 > 25 and y2 - y1 > 20:
                return image[y1:y2, x1:x2], pair["rgb"]
    raise ValueError("No visible pallet crops found in training subset")


def generate_video(config: Config, index: int, scenario: str | None = None) -> Path:
    settings = config.vision.video
    width, height = settings.width, settings.height
    scenario = scenario or SCENARIOS[index % len(SCENARIOS)]
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    directory = config.paths.data / "synthetic/tracking"
    directory.mkdir(parents=True, exist_ok=True)
    video_id = f"VIDEO_{index:03d}"
    path = directory / f"{video_id}.mp4"
    frames = settings.seconds * settings.fps
    count = 4 if scenario == "crowded_crossing" else (1 if scenario == "reverse_direction" else 2)
    line = [[width / 2, 0], [width / 2, height]]
    zone = [
        [width * 0.52, 30],
        [width * 0.9, 30],
        [width * 0.9, height - 30],
        [width * 0.52, height - 30],
    ]
    crop, source = (
        (None, "locally_drawn_fixture") if config.mode == "smoke" else _public_crop(config)
    )
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), settings.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder could not be opened")
    records, crossings = [], []
    previous: dict[int, tuple[float, float]] = {}
    crossed: set[int] = set()
    try:
        for frame_index in range(frames):
            progress = frame_index / max(frames - 1, 1)
            image = np.full((height, width, 3), (43, 35, 29), dtype=np.uint8)
            for grid_x in range(0, width, 80):
                cv2.line(image, (grid_x, 0), (grid_x, height), (56, 47, 39), 1)
            for grid_y in range(0, height, 80):
                cv2.line(image, (0, grid_y), (width, grid_y), (56, 47, 39), 1)
            cv2.line(image, (width // 2, 0), (width // 2, height), (195, 220, 216), 2)
            objects = []
            for track_id in range(1, count + 1):
                reverse = scenario == "reverse_direction"
                t = 1 - progress if reverse else progress
                if scenario == "partial_exit":
                    t *= 0.43
                size = 78 + int(8 * math.sin(progress * math.pi))
                x = int(10 + (width - size - 20) * t - (track_id - 1) * 26)
                y = int(height * track_id / (count + 1) - 24)
                if scenario in ["two_pallets_close", "duplicate_visual_overlap"]:
                    y = (
                        height // 2
                        - 25
                        + (track_id - 1) * (18 if scenario == "duplicate_visual_overlap" else 59)
                    )
                x = max(0, min(width - size, x))
                box = [x, y, x + size, y + 48]
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if track_id in previous and track_id not in crossed:
                    if crosses_segment(previous[track_id], center, line, 1):
                        crossings.append({"track_id": track_id, "frame_index": frame_index})
                        crossed.add(track_id)
                if side(center, line) != 0:
                    previous[track_id] = center
                occluded = (
                    scenario == "temporary_occlusion"
                    and track_id == 1
                    and frames // 3 <= frame_index < frames // 3 + 4
                )
                objects.append({"track_id": track_id, "bbox": box, "visible": not occluded})
                if occluded:
                    continue
                if crop is None:
                    cv2.rectangle(image, (x, y), (x + size, y + 48), (72, 139, 199), -1)
                    for offset in [9, 24, 39]:
                        cv2.line(
                            image, (x + 2, y + offset), (x + size - 2, y + offset), (43, 85, 129), 3
                        )
                else:
                    scaled = cv2.resize(crop, (size, 48))
                    image[y : y + 48, x : x + size] = scaled
            cv2.putText(
                image,
                "CONTROLLED SYNTHETIC SEQUENCE",
                (18, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (212, 219, 222),
                1,
            )
            writer.write(image)
            records.append({"frame_index": frame_index, "objects": objects})
    finally:
        writer.release()
    write_json(
        path.with_suffix(".json"),
        {
            "video_id": video_id,
            "scenario": scenario,
            "seed": config.seed + index,
            "fps": settings.fps,
            "width": width,
            "height": height,
            "frames": records,
            "line": line,
            "zone": zone,
            "crossings": crossings,
            "expected_unique_count": len(crossed),
            "source": source,
            "trajectory": "deterministic horizontal motion with scenario-specific visibility",
        },
    )
    return path


def generate_videos(config: Config) -> list[Path]:
    return [generate_video(config, index) for index in range(config.vision.video.count)]
