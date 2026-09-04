from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
DocumentType = Literal["bill_of_lading", "advance_shipping_notice", "invoice", "packing_list"]


class ShipmentDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1)
    document_type: DocumentType
    shipment_id: str | None = None
    carrier: str | None = None
    vehicle_id: str | None = None
    origin: str | None = None
    destination: str | None = None
    pallet_count: int | None = Field(None, ge=0)
    package_count: int | None = Field(None, ge=0)
    total_weight_lb: float | None = Field(None, ge=0, allow_inf_nan=False)
    extraction_confidence: Confidence = 0
    field_confidences: dict[str, Confidence] = Field(default_factory=dict)
