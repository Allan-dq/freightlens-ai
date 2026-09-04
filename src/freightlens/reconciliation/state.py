from freightlens.schemas.document import ShipmentDocument
from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState


def expected_from_document(document: ShipmentDocument) -> ExpectedShipmentState:
    if not document.shipment_id:
        raise ValueError("Document has no shipment ID; manual association is required")
    return ExpectedShipmentState(
        shipment_id=document.shipment_id,
        expected_pallet_count=document.pallet_count,
        expected_vehicle_id=document.vehicle_id,
        document_confidence=min(
            document.extraction_confidence, document.field_confidences.get("pallet_count", 0)
        ),
    )


def observed_from_tracking(
    result: dict, shipment_id: str, vehicle_id: str | None = None
) -> ObservedShipmentState:
    detections = [event for event in result["events"] if event["event_type"] == "detected"]
    crossing_ids = {
        event["track_id"] for event in result["events"] if event["event_type"] == "crossed_line"
    }
    times = [event["timestamp_sec"] for event in detections]
    confidences = [event["confidence"] for event in detections]
    return ObservedShipmentState(
        shipment_id=shipment_id,
        observed_pallet_count=len(crossing_ids) if result.get("complete_observation") else None,
        observed_vehicle_id=vehicle_id,
        first_activity_sec=min(times) if times else None,
        last_activity_sec=max(times) if times else None,
        mean_detection_confidence=sum(confidences) / len(confidences) if confidences else None,
        # Identity fragmentation cannot be inferred reliably without tracking ground truth.
        tracking_fragmentation=None,
    )
