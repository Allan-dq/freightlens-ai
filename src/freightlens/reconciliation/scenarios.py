import random

from freightlens.config import Config
from freightlens.io import read_json, write_json

KINDS = [
    "NORMAL",
    "MISSING_PALLET",
    "EXTRA_PALLET",
    "COUNT_MISMATCH",
    "VEHICLE_MISMATCH",
    "LOW_CONFIDENCE_DOCUMENT",
    "LOW_CONFIDENCE_VISION",
    "PARTIAL_OBSERVATION",
]


def generate_scenarios(config: Config) -> list[dict]:
    rng = random.Random(config.seed)
    documents = sorted((config.paths.data / "synthetic/documents/test").glob("DOC_*.json"))
    tracks = [
        config.paths.artifacts / "predictions" / f"VIDEO_{i:03d}_tracking.json"
        for i in range(config.vision.video.count)
    ]
    if not documents or not tracks or not all(path.exists() for path in tracks):
        raise ValueError("Generate documents and run tracking before building scenarios")
    kinds = [KINDS[index % len(KINDS)] for index in range(config.reconciliation.scenarios)]
    rng.shuffle(kinds)
    result = []
    for index, kind in enumerate(kinds):
        document = read_json(documents[index % len(documents)])
        tracking = read_json(tracks[index % len(tracks)])
        count = max(document["fields"]["pallet_count"], 1)
        # Controlled interventions align independently generated visual events to a shipment.
        # Preserve the original visual count and every intervention for auditing.
        base_count = tracking["unique_count"]
        shipment_id = f"CASE-{index:06d}"
        vehicle = document["fields"].get("vehicle_id") or "TR-1000"
        expected = {
            "shipment_id": shipment_id,
            "expected_pallet_count": count,
            "expected_vehicle_id": vehicle,
            "document_confidence": 0.95,
        }
        observed = {
            "shipment_id": shipment_id,
            "observed_pallet_count": count,
            "observed_vehicle_id": vehicle,
            "mean_detection_confidence": 0.9,
            "tracking_fragmentation": 0.0,
        }
        anomaly = kind in ["MISSING_PALLET", "EXTRA_PALLET", "COUNT_MISMATCH", "VEHICLE_MISMATCH"]
        noise = "none"
        if kind == "MISSING_PALLET":
            observed["observed_pallet_count"] = count - 1
            noise = "detector_undercount_or_missing_pallet"
        elif kind == "EXTRA_PALLET":
            observed["observed_pallet_count"] = count + 1
            noise = "detector_overcount_or_extra_pallet"
        elif kind == "COUNT_MISMATCH":
            observed["observed_pallet_count"] = count + 2
            observed["observed_vehicle_id"] = "OTHER-999"
        elif kind == "VEHICLE_MISMATCH":
            observed["observed_vehicle_id"] = "OTHER-999"
        elif kind == "LOW_CONFIDENCE_DOCUMENT":
            expected["document_confidence"] = 0.3
            expected["expected_pallet_count"] = None
            noise = "ocr_field_missing"
        elif kind == "LOW_CONFIDENCE_VISION":
            observed["mean_detection_confidence"] = 0.25
            observed["observed_vehicle_id"] = None
            noise = "low_confidence_unknown_vehicle"
        elif kind == "PARTIAL_OBSERVATION":
            observed["observed_pallet_count"] = None
            observed["tracking_fragmentation"] = 0.7
            noise = "tracking_fragmentation"
        # Independent hidden upstream errors create honest rule failure cases.
        if kind == "NORMAL" and index % 2 == 0:
            expected["expected_pallet_count"] = count + 1
            noise = "confident_ocr_wrong_count"
        result.append(
            {
                "id": shipment_id,
                "expected": expected,
                "observed": observed,
                "ground_truth": {
                    "anomaly": anomaly,
                    "anomaly_type": kind,
                    "review_required": kind.startswith("LOW_") or kind == "PARTIAL_OBSERVATION",
                },
                "provenance": {
                    "document": document["id"],
                    "video": tracking["video_id"],
                    "original_visual_count": base_count,
                    "aligned_count": count,
                    "noise": noise,
                    "source": "controlled_state_intervention",
                },
            }
        )
    write_json(config.paths.data / "synthetic/scenarios/scenarios.json", result)
    return result
