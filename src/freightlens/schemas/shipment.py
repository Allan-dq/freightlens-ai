from pydantic import BaseModel, Field, model_validator

from freightlens.schemas.document import Confidence


class ExpectedShipmentState(BaseModel):
    shipment_id: str = Field(min_length=1)
    expected_pallet_count: int | None = Field(None, ge=0)
    expected_vehicle_id: str | None = None
    document_confidence: Confidence = 0


class ObservedShipmentState(BaseModel):
    shipment_id: str = Field(min_length=1)
    observed_pallet_count: int | None = Field(None, ge=0)
    observed_vehicle_id: str | None = None
    first_activity_sec: float | None = Field(None, ge=0, allow_inf_nan=False)
    last_activity_sec: float | None = Field(None, ge=0, allow_inf_nan=False)
    mean_detection_confidence: Confidence | None = None
    tracking_fragmentation: Confidence | None = None

    @model_validator(mode="after")
    def ordered_time(self):
        if (
            self.first_activity_sec is not None
            and self.last_activity_sec is not None
            and self.first_activity_sec > self.last_activity_sec
        ):
            raise ValueError("Last activity precedes first activity")
        return self
