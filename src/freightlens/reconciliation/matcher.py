from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState


def match_shipment(expected: ExpectedShipmentState, observed: ObservedShipmentState) -> None:
    if expected.shipment_id != observed.shipment_id:
        raise ValueError("Shipment IDs differ; refusing to reconcile unrelated observations")
