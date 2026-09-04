from typing import Literal

from pydantic import BaseModel, Field, field_validator

from freightlens.schemas.document import Confidence


class DetectionEvent(BaseModel):
    video_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0, allow_inf_nan=False)
    track_id: int = Field(ge=0)
    object_class: str = "pallet"
    confidence: Confidence
    bbox_xyxy: tuple[float, float, float, float]
    event_type: Literal["detected", "entered_zone", "crossed_line", "exited_zone"] | None = None

    @field_validator("bbox_xyxy")
    @classmethod
    def valid_box(cls, box):
        import math

        if not all(math.isfinite(v) for v in box):
            raise ValueError("Box coordinates must be finite")
        if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Expected positive xyxy box")
        return box
