def score_evidence(
    delta: int | None,
    expected_count: int | None,
    vehicle_match: bool | None,
    document_confidence: float,
    vision_confidence: float | None,
    fragmentation: float | None,
) -> dict:
    components = {
        "count_disagreement": min(abs(delta or 0) / max(expected_count or 0, 1), 1),
        "vehicle_disagreement": float(vehicle_match is False),
        "document_uncertainty": 1 - document_confidence,
        "vision_uncertainty": 1 - (vision_confidence or 0),
        "fragmentation": fragmentation or 0,
    }
    weights = {
        "count_disagreement": 0.45,
        "vehicle_disagreement": 0.3,
        "document_uncertainty": 0.1,
        "vision_uncertainty": 0.1,
        "fragmentation": 0.05,
    }
    return {
        "score": min(1.0, sum(components[key] * weight for key, weight in weights.items())),
        "components": components,
        "weights": weights,
        "calibrated_probability": False,
    }
