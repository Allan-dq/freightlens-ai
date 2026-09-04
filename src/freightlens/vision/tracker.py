"""Per-video tracker state, fixture pixel detections, and real ByteTrack integration."""

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from freightlens.config import Config
from freightlens.io import write_json
from freightlens.schemas.vision import DetectionEvent
from freightlens.vision.counting import CrossingCounter


class FixturePixelTracker:
    """A deterministic color/centroid test double; never reads ground-truth boxes or IDs."""

    def __init__(self):
        self.tracks: dict[int, tuple[np.ndarray, int]] = {}
        self.next_id = 1

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([8, 70, 80]), np.array([35, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= 300:
                boxes.append([x, y, x + w, y + h])
        return sorted(boxes)

    def update(self, frame, frame_index):
        boxes = self.detect(frame)
        self.tracks = {
            key: value for key, value in self.tracks.items() if frame_index - value[1] <= 6
        }
        ids = list(self.tracks)
        assignment = {}
        centers = [np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) for b in boxes]
        if ids and boxes:
            distance = np.array(
                [
                    [np.linalg.norm(self.tracks[key][0] - center) for center in centers]
                    for key in ids
                ]
            )
            rows, cols = linear_sum_assignment(distance)
            for row, col in zip(rows, cols, strict=True):
                if distance[row, col] < 110:
                    assignment[col] = ids[row]
        result = []
        for index, (box, center) in enumerate(zip(boxes, centers, strict=True)):
            if index not in assignment:
                assignment[index] = self.next_id
                self.next_id += 1
            key = assignment[index]
            self.tracks[key] = center, frame_index
            result.append((key, box, 0.9))
        return result


class ByteTrackBackend:
    def __init__(self, config: Config):
        from freightlens.vision.detector import load_detector, pallet_class, trained_checkpoint

        self.model = load_detector(trained_checkpoint(config))
        self.class_id = pallet_class(self.model)
        self.config = config

    def update(self, frame, frame_index):
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device="cpu",
            classes=[self.class_id],
            conf=self.config.vision.tracker.confidence_threshold,
            imgsz=self.config.vision.detector.image_size,
            verbose=False,
        )[0]
        if result.boxes.id is None:
            return []
        return [
            (int(track), box.tolist(), float(confidence))
            for track, box, confidence in zip(
                result.boxes.id.cpu(), result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), strict=True
            )
        ]


def track_video(
    path: Path, config: Config, *, line=None, zone=None, output: Path | None = None
) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Unreadable video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if fps <= 0 or width * height <= 0 or width * height > config.api.max_video_pixels:
        capture.release()
        raise ValueError("Invalid video metadata or dimensions")
    if capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps > config.api.max_video_seconds:
        capture.release()
        raise ValueError("Video exceeds duration limit")
    line = line or [[width / 2, 0], [width / 2, height]]
    zone = zone or [
        [width * 0.52, 30],
        [width * 0.9, 30],
        [width * 0.9, height - 30],
        [width * 0.52, height - 30],
    ]
    counter = CrossingCounter(
        line, zone, config.vision.tracker.direction, config.vision.tracker.confidence_threshold
    )
    backend = (
        FixturePixelTracker()
        if config.vision.tracker.type == "fixture"
        else ByteTrackBackend(config)
    )
    output = output or config.paths.artifacts / "videos" / f"{path.stem}_annotated.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot encode tracking video")
    events: list[DetectionEvent] = []
    lifecycle: dict[str, dict[str, int]] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index / fps > config.api.max_video_seconds:
                raise ValueError("Decoded video exceeds duration limit")
            for track_id, box, confidence in backend.update(frame, frame_index):
                box = [
                    max(0, float(box[0])),
                    max(0, float(box[1])),
                    min(width, float(box[2])),
                    min(height, float(box[3])),
                ]
                detection = DetectionEvent(
                    video_id=path.stem,
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    track_id=track_id,
                    confidence=confidence,
                    bbox_xyxy=tuple(box),
                )
                events.extend(counter.update(detection))
                lifecycle.setdefault(str(track_id), {"first_frame": frame_index})
                lifecycle[str(track_id)]["last_frame"] = frame_index
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 232, 88), 2)
                cv2.putText(
                    frame,
                    f"Pallet #{track_id}",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (140, 232, 88),
                    1,
                )
            cv2.putText(
                frame,
                f"UNIQUE CROSSINGS  {len(counter.counted)}",
                (20, height - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (221, 238, 245),
                2,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if frame_index == 0:
        raise ValueError("Video contains no decodable frames")
    result = {
        "video_id": path.stem,
        "backend": config.vision.tracker.type,
        "fixture": config.vision.tracker.type == "fixture",
        "unique_count": len(counter.counted),
        "frames": frame_index,
        "events": [event.model_dump() for event in events],
        "lifecycle": lifecycle,
        "annotated_video": str(output),
        "complete_observation": True,
    }
    write_json(config.paths.artifacts / "predictions" / f"{path.stem}_tracking.json", result)
    return result
