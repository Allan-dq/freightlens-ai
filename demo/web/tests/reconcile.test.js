// Checks demo/web/reconcile.js against the cases generated from the Python rules.
// Run: node demo/web/tests/reconcile.test.js
const assert = require("node:assert/strict");
const path = require("node:path");
const { reconcile } = require("../reconcile.js");
const cases = require(path.join(__dirname, "reconcile_cases.json"));

for (const testCase of cases) {
  const result = reconcile(testCase.expected, testCase.observed);
  const projected = {
    status: result.status,
    anomaly_type: result.anomaly_type,
    pallet_delta: result.pallet_delta,
    vehicle_match: result.vehicle_match,
    anomaly_score: Math.round(result.anomaly_score * 1e6) / 1e6,
    review_reasons: result.evidence.review_reasons,
    discrepancies: result.evidence.discrepancies,
  };
  assert.deepEqual(projected, testCase.result, testCase.name);
}
console.log(`${cases.length} reconciliation cases match the Python rules`);
