// Port of freightlens.reconciliation.rules.reconcile and scoring.score_evidence.
// demo/web/tests/reconcile_cases.json holds shared cases: tests/unit/test_web_reconcile.py
// checks them against Python and demo/web/tests/reconcile.test.js against this file.

const DEFAULTS = { document_threshold: 0.7, vision_threshold: 0.6, fragmentation_threshold: 0.4 };
const WEIGHTS = {
  count_disagreement: 0.45,
  vehicle_disagreement: 0.3,
  document_uncertainty: 0.1,
  vision_uncertainty: 0.1,
  fragmentation: 0.05,
};

function normalizeVehicle(value) {
  if (value === null || value === undefined) return null;
  const cleaned = String(value).normalize("NFKC").toUpperCase().replace(/[^A-Z0-9]/g, "");
  return cleaned || null;
}

function scoreEvidence(delta, expectedCount, vehicleMatch, documentConfidence, visionConfidence, fragmentation) {
  const components = {
    count_disagreement: Math.min(Math.abs(delta ?? 0) / Math.max(expectedCount ?? 0, 1), 1),
    vehicle_disagreement: vehicleMatch === false ? 1 : 0,
    document_uncertainty: 1 - documentConfidence,
    vision_uncertainty: 1 - (visionConfidence ?? 0),
    fragmentation: fragmentation ?? 0,
  };
  let score = 0;
  for (const [key, weight] of Object.entries(WEIGHTS)) score += components[key] * weight;
  return { score: Math.min(1, score), components, weights: WEIGHTS };
}

function reconcile(expected, observed, config = DEFAULTS) {
  if (expected.shipment_id !== observed.shipment_id) throw new Error("Shipment IDs differ");
  const e = expected.expected_pallet_count ?? null;
  const o = observed.observed_pallet_count ?? null;
  const delta = e !== null && o !== null ? o - e : null;
  const ev = normalizeVehicle(expected.expected_vehicle_id);
  const ov = normalizeVehicle(observed.observed_vehicle_id);
  const vehicleMatch = ev !== null && ov !== null ? ev === ov : null;
  const review = [];
  if (expected.document_confidence < config.document_threshold) review.push("LOW_CONFIDENCE_DOCUMENT");
  if (observed.mean_detection_confidence == null || observed.mean_detection_confidence < config.vision_threshold) {
    review.push("LOW_CONFIDENCE_VISION");
  }
  if (
    delta === null ||
    (observed.tracking_fragmentation != null && observed.tracking_fragmentation > config.fragmentation_threshold)
  ) {
    review.push("PARTIAL_OBSERVATION");
  }
  const anomalies = [];
  if (delta !== null && delta !== 0) anomalies.push(delta < 0 ? "MISSING_PALLET" : "EXTRA_PALLET");
  if (vehicleMatch === false) anomalies.push("VEHICLE_MISMATCH");
  const status = review.length ? "REVIEW_REQUIRED" : anomalies.length ? "ANOMALY" : "MATCH";
  const kind = review.length
    ? review[0]
    : anomalies.length > 1
      ? "COUNT_MISMATCH"
      : anomalies[0] || "NORMAL";
  const scoring = scoreEvidence(
    delta,
    e,
    vehicleMatch,
    expected.document_confidence,
    observed.mean_detection_confidence,
    observed.tracking_fragmentation,
  );
  return {
    shipment_id: expected.shipment_id,
    expected_pallet_count: e,
    observed_pallet_count: o,
    pallet_delta: delta,
    vehicle_match: vehicleMatch,
    anomaly: status === "ANOMALY",
    anomaly_type: kind,
    anomaly_score: scoring.score,
    status,
    evidence: { review_reasons: review, discrepancies: anomalies, score_details: scoring },
  };
}

if (typeof module !== "undefined") module.exports = { reconcile, normalizeVehicle, scoreEvidence };
