"""Shared reconciliation cases for the Python service and the browser demo.

The browser demo re-implements the reconciliation rules in demo/web/reconcile.js. Both
implementations are checked against demo/web/tests/reconcile_cases.json: this test for
Python, demo/web/tests/reconcile.test.js (run in CI with Node) for JavaScript.
Regenerate the file with: python tests/unit/test_web_reconcile.py
"""

import json
from pathlib import Path

import pytest

from freightlens.reconciliation.rules import reconcile
from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState

CASES_PATH = Path(__file__).parents[2] / "demo/web/tests/reconcile_cases.json"

INPUTS = [
    ("match", 6, "TR-1234", 0.95, 6, "tr 1234", 0.8, None),
    ("missing pallets", 6, "TR-1234", 0.95, 4, None, 0.8, None),
    ("extra pallet", 3, None, 0.9, 4, None, 0.75, None),
    ("low vision confidence", 6, "TR-1234", 0.95, 6, None, 0.45, None),
    ("low document confidence", 6, None, 0.5, 5, None, 0.9, None),
    ("vehicle mismatch", 2, "TR-1111", 0.9, 2, "TR-2222", 0.9, None),
    ("count and vehicle", 5, "TR-1111", 0.9, 3, "TR-2222", 0.9, None),
    ("no observation", 5, None, 0.9, None, None, None, None),
    ("fragmented tracks", 5, None, 0.9, 5, None, 0.9, 0.6),
]


def _project(result: dict) -> dict:
    return {
        "status": result["status"],
        "anomaly_type": result["anomaly_type"],
        "pallet_delta": result["pallet_delta"],
        "vehicle_match": result["vehicle_match"],
        "anomaly_score": round(result["anomaly_score"], 6),
        "review_reasons": result["evidence"]["review_reasons"],
        "discrepancies": result["evidence"]["discrepancies"],
    }


def build_cases() -> list[dict]:
    cases = []
    for name, e, ev, dc, o, ov, vc, frag in INPUTS:
        expected = {
            "shipment_id": "SHP-1",
            "expected_pallet_count": e,
            "expected_vehicle_id": ev,
            "document_confidence": dc,
        }
        observed = {
            "shipment_id": "SHP-1",
            "observed_pallet_count": o,
            "observed_vehicle_id": ov,
            "mean_detection_confidence": vc,
            "tracking_fragmentation": frag,
        }
        result = reconcile(
            ExpectedShipmentState.model_validate(expected),
            ObservedShipmentState.model_validate(observed),
        ).model_dump()
        cases.append(
            {"name": name, "expected": expected, "observed": observed, "result": _project(result)}
        )
    return cases


SHARED_CASES = json.loads(CASES_PATH.read_text(encoding="utf-8")) if CASES_PATH.exists() else []


def test_shared_cases_cover_every_input():
    assert [c["name"] for c in SHARED_CASES] == [row[0] for row in INPUTS]


@pytest.mark.parametrize("case", SHARED_CASES, ids=lambda c: c["name"])
def test_shared_cases_match_python_rules(case):
    current = {c["name"]: c for c in build_cases()}[case["name"]]
    assert current == case


if __name__ == "__main__":
    CASES_PATH.write_text(json.dumps(build_cases(), indent=2) + "\n", encoding="utf-8")
