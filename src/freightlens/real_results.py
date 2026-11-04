"""Render the real-data results table from saved metric JSON.

Only measured values are rendered; a stage without metrics is listed as "not run".
The table replaces the block between REAL_RESULTS markers in README.md.
"""

from pathlib import Path

from freightlens.config import Config
from freightlens.io import read_json

START, END = "<!-- REAL_RESULTS_START -->", "<!-- REAL_RESULTS_END -->"
LATENCY_MODELS = {
    "detector_yolo11n_640": "YOLO11n, 640 px input",
    "layoutlmv3_base_512_tokens": "LayoutLMv3-base, 512 tokens (OCR excluded)",
}


def _load(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "not run" if value is None else f"{value:.{digits}f}"


def results_table(config: Config) -> str:
    metrics = config.paths.artifacts / "metrics"
    rows = [
        "| Model | Trained on | Evaluated on | Metric | Value |",
        "|---|---|---|---|---|",
    ]
    detection = _load(metrics / "pallet_detection.json")
    loco = _load(config.paths.data / "manifests/loco.json")
    if detection and detection.get("dataset") == "loco" and loco:
        splits = loco["splits"]
        trained = f"LOCO, {splits['train']['images']:,} real warehouse images"
        tested = f"LOCO, {splits['test']['images']:,} images from unseen recording sessions"
        rows += [
            f"| YOLO11n detector | {trained} | {tested} | pallet AP50 | "
            f"{_fmt(detection['pallet_AP50'])} |",
            f"| | | | pallet AP50-95 | {_fmt(detection['pallet_AP50_95'])} |",
            f"| | | | mAP50, all 5 classes | {_fmt(detection['mAP50'])} |",
        ]
        calibration = _load(metrics / "pallet_count_calibration.json")
        if calibration:
            rows.append(
                f"| | | | pallet count error per image, threshold "
                f"{calibration['threshold']:.2f} chosen on validation (MAE / bias) | "
                f"{calibration['test_mae']:.1f} / {calibration['test_bias']:+.1f} |"
            )
        else:
            rows.append(
                f"| | | | pallet count error per image (MAE) | "
                f"{_fmt(detection['pallet_count_mae'], 1)} |"
            )
    else:
        rows.append("| YOLO11n detector | LOCO | LOCO | pallet AP50 | not run |")
    for suffix, label in [("", "word boxes (OCR setting)"), ("-segment", "segment boxes")]:
        funsd = _load(metrics / f"document_layoutlm_funsd{suffix}.json")
        split = funsd["split"] if funsd else {"train": "-", "test": "-"}
        rows.append(
            f"| LayoutLMv3-base, {label} | FUNSD, {split['train']} scanned forms | "
            f"FUNSD official test, {split['test']} forms | entity F1 | "
            f"{_fmt(funsd['f1'] if funsd else None)} |"
        )
    latency = _load(metrics / "latency.json")
    if latency:
        devices = sorted({d for key in LATENCY_MODELS for d in latency.get(key, {})})
        rows += [
            "",
            f"Median single-image latency on {latency['hardware']}:",
            "",
            "| Model | " + " | ".join(d.upper() for d in devices) + " |",
            "|---|" + "---|" * len(devices),
        ]
        for key, label in LATENCY_MODELS.items():
            values = latency.get(key, {})
            cells = [f"{values[d]['median_ms']:.0f} ms" if d in values else "-" for d in devices]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def update_readme(config: Config, readme: Path = Path("README.md")) -> dict:
    table = results_table(config)
    source = readme.read_text(encoding="utf-8")
    if START not in source or END not in source:
        raise ValueError(f"{readme} has no real-results markers")
    head, rest = source.split(START, 1)
    readme.write_text(head + START + "\n" + table + "\n" + END + rest.split(END, 1)[1])
    return {"readme": str(readme), "rows": table.count("\n") - 1}
