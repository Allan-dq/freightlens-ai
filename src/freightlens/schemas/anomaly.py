from typing import Literal

from pydantic import BaseModel, Field

from freightlens.schemas.document import Confidence


class ReconciliationResult(BaseModel):
    shipment_id: str
    expected_pallet_count: int | None
    observed_pallet_count: int | None
    pallet_delta: int | None
    expected_vehicle_id: str | None
    observed_vehicle_id: str | None
    vehicle_match: bool | None
    anomaly: bool
    anomaly_type: str | None
    anomaly_score: Confidence
    evidence: dict[str, object] = Field(default_factory=dict)
    status: Literal["MATCH", "ANOMALY", "REVIEW_REQUIRED"]
