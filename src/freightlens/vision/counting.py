from freightlens.schemas.vision import DetectionEvent
from freightlens.vision.geometry import crosses_segment, point_in_polygon, side


class CrossingCounter:
    def __init__(self, line, zone, direction: int = 1, threshold: float = 0.25):
        if line[0] == line[1]:
            raise ValueError("Counting line must have nonzero length")
        self.line, self.zone = line, zone
        self.direction, self.threshold = direction, threshold
        self.positions: dict[int, tuple[float, float]] = {}
        self.in_zone: dict[int, bool] = {}
        self.counted: set[int] = set()

    def update(self, detection: DetectionEvent) -> list[DetectionEvent]:
        if detection.confidence < self.threshold:
            return []
        box = detection.bbox_xyxy
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        track = detection.track_id
        events = [detection.model_copy(update={"event_type": "detected"})]
        previous = self.positions.get(track)
        if previous is not None and track not in self.counted:
            if crosses_segment(previous, center, self.line, self.direction):
                events.append(detection.model_copy(update={"event_type": "crossed_line"}))
                self.counted.add(track)
        # Retain the last nonzero side when an object is exactly on the line.
        if side(center, self.line) != 0:
            self.positions[track] = center
        inside = point_in_polygon(center, self.zone)
        was_inside = self.in_zone.get(track, False)
        if inside != was_inside:
            events.append(
                detection.model_copy(
                    update={"event_type": "entered_zone" if inside else "exited_zone"}
                )
            )
        self.in_zone[track] = inside
        return events
