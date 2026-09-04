from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from freightlens.config import Config
from freightlens.io import read_json, record_run, write_json
from freightlens.reconciliation.rules import reconcile
from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState


def run_reconciliation(config: Config) -> dict:
    scenarios = read_json(config.paths.data / "synthetic/scenarios/scenarios.json")
    results = [
        reconcile(
            ExpectedShipmentState.model_validate(case["expected"]),
            ObservedShipmentState.model_validate(case["observed"]),
            config.reconciliation,
        ).model_dump()
        for case in scenarios
    ]
    truth = [case["ground_truth"]["anomaly"] for case in scenarios]
    guesses = [result["anomaly"] for result in results]
    scores = [result["anomaly_score"] for result in results]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, guesses, average="binary", zero_division=0
    )
    types = classification_report(
        [c["ground_truth"]["anomaly_type"] for c in scenarios],
        [r["anomaly_type"] for r in results],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "status": "measured",
        "evaluation_scope": "controlled_state_interventions",
        "samples": len(scenarios),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(truth, guesses, labels=[False, True]).tolist(),
        "confusion_matrix_labels": ["normal_or_review", "anomaly"],
        "roc_auc": float(roc_auc_score(truth, scores)) if len(set(truth)) > 1 else None,
        "pr_auc": float(average_precision_score(truth, scores)) if any(truth) else None,
        "macro_f1": types["macro avg"]["f1-score"],
        "per_class": types,
        "review_rate": sum(r["status"] == "REVIEW_REQUIRED" for r in results) / len(results),
        "review_accuracy": sum(
            (r["status"] == "REVIEW_REQUIRED") == c["ground_truth"]["review_required"]
            for r, c in zip(results, scenarios, strict=True)
        )
        / len(results),
    }
    errors = [
        {"scenario": case, "prediction": result}
        for case, result in zip(scenarios, results, strict=True)
        if case["ground_truth"]["anomaly_type"] != result["anomaly_type"]
    ]
    write_json(config.paths.artifacts / "metrics/reconciliation.json", metrics)
    write_json(config.paths.artifacts / "predictions/reconciliation.json", results)
    write_json(config.paths.artifacts / "predictions/reconciliation_errors.json", errors)
    record_run(config, "reconciliation", metrics)
    return metrics
