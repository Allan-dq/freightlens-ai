from freightlens.config import ReconciliationConfig
from freightlens.document_ai.normalize import normalize_vehicle
from freightlens.reconciliation.matcher import match_shipment
from freightlens.reconciliation.scoring import score_evidence
from freightlens.schemas.anomaly import ReconciliationResult
from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState


def reconcile(
    expected: ExpectedShipmentState,
    observed: ObservedShipmentState,
    config: ReconciliationConfig | None = None,
) -> ReconciliationResult:
    match_shipment(expected, observed)
    config = config or ReconciliationConfig()
    e, o = expected.expected_pallet_count, observed.observed_pallet_count
    delta = o - e if o is not None and e is not None else None
    ev, ov = (
        normalize_vehicle(expected.expected_vehicle_id),
        normalize_vehicle(observed.observed_vehicle_id),
    )
    vehicle_match = ev == ov if ev is not None and ov is not None else None
    review = []
    if expected.document_confidence < config.document_threshold:
        review.append("LOW_CONFIDENCE_DOCUMENT")
    if (
        observed.mean_detection_confidence is None
        or observed.mean_detection_confidence < config.vision_threshold
    ):
        review.append("LOW_CONFIDENCE_VISION")
    if delta is None or (
        observed.tracking_fragmentation is not None
        and observed.tracking_fragmentation > config.fragmentation_threshold
    ):
        review.append("PARTIAL_OBSERVATION")
    anomalies = []
    if delta is not None and delta != 0:
        anomalies.append("MISSING_PALLET" if delta < 0 else "EXTRA_PALLET")
    if vehicle_match is False:
        anomalies.append("VEHICLE_MISMATCH")
    status = "REVIEW_REQUIRED" if review else ("ANOMALY" if anomalies else "MATCH")
    kind = (
        review[0]
        if review
        else ("COUNT_MISMATCH" if len(anomalies) > 1 else (anomalies[0] if anomalies else "NORMAL"))
    )
    scoring = score_evidence(
        delta,
        e,
        vehicle_match,
        expected.document_confidence,
        observed.mean_detection_confidence,
        observed.tracking_fragmentation,
    )
    return ReconciliationResult.model_validate(
        {
            "shipment_id": expected.shipment_id,
            "expected_pallet_count": e,
            "observed_pallet_count": o,
            "pallet_delta": delta,
            "expected_vehicle_id": expected.expected_vehicle_id,
            "observed_vehicle_id": observed.observed_vehicle_id,
            "vehicle_match": vehicle_match,
            "anomaly": status == "ANOMALY",
            "anomaly_type": kind,
            "anomaly_score": scoring["score"],
            "status": status,
            "evidence": {
                "review_reasons": review,
                "discrepancies": anomalies,
                "vehicle_verified": vehicle_match is not None,
                "score_details": scoring,
                "document_confidence": expected.document_confidence,
                "vision_confidence": observed.mean_detection_confidence,
            },
        }
    )
