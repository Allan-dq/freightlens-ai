import pytest

from freightlens.reconciliation.rules import reconcile
from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState


def states(count=8, confidence=0.9):
    return (
        ExpectedShipmentState(
            shipment_id="S1",
            expected_pallet_count=8,
            expected_vehicle_id="TR-12",
            document_confidence=0.95,
        ),
        ObservedShipmentState(
            shipment_id="S1",
            observed_pallet_count=count,
            observed_vehicle_id="tr 12",
            mean_detection_confidence=confidence,
        ),
    )


@pytest.mark.parametrize("count,kind", [(8, "NORMAL"), (7, "MISSING_PALLET"), (9, "EXTRA_PALLET")])
def test_count_rules(count, kind):
    expected, observed = states(count)
    result = reconcile(expected, observed)
    assert result.anomaly_type == kind
    assert result.pallet_delta == count - 8
    assert result.vehicle_match


def test_low_confidence_abstains_despite_disagreement():
    expected, observed = states(2, 0.3)
    result = reconcile(expected, observed)
    assert result.status == "REVIEW_REQUIRED"
    assert not result.anomaly
    assert "MISSING_PALLET" in result.evidence["discrepancies"]


def test_unknown_is_not_zero():
    expected, observed = states(None)
    result = reconcile(expected, observed)
    assert result.pallet_delta is None
    assert result.anomaly_type == "PARTIAL_OBSERVATION"


def test_unrelated_shipments_rejected():
    expected, observed = states()
    observed.shipment_id = "OTHER"
    with pytest.raises(ValueError, match="Shipment IDs"):
        reconcile(expected, observed)


def test_multiple_discrepancies_and_unknown_plate():
    expected, observed = states(7)
    observed.observed_vehicle_id = "TR-99"
    result = reconcile(expected, observed)
    assert result.anomaly_type == "COUNT_MISMATCH"
    observed.observed_vehicle_id = None
    assert reconcile(expected, observed).vehicle_match is None
