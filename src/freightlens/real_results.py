"""Render the real-data results tables from saved metric JSON.

Only measured values are rendered; a stage without metrics is listed as "not run".
The tables replace the block between REAL_RESULTS markers in README.md.

The detector table compares the final model (trained on every non-test LOCO session for a
fixed number of epochs, configs/real_final.yaml) with the development model it was derived
from (configs/real.yaml, epoch count selected on a held-out session). Both are scored on the
same test sessions.
"""

from pathlib import Path

from freightlens.config import Config, load_config
from freightlens.io import read_json

START, END = "<!-- REAL_RESULTS_START -->", "<!-- REAL_RESULTS_END -->"
FINAL_CONFIG = Path("configs/real_final.yaml")
LATENCY_MODELS = {
    "detector_yolo11n_640": "YOLO11n, 640 px input",
    "layoutlmv3_base_512_tokens": "LayoutLMv3-base, 512 tokens (OCR excluded)",
}


def _load(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "not run" if value is None else f"{value:.{digits}f}"


def _sessions(subsets: list[int]) -> str:
    return (
        ", ".join(map(str, subsets[:-1])) + (" and " if len(subsets) > 1 else "") + str(subsets[-1])
    )


def _detector_row(label: str, config: Config, training: str) -> str | None:
    metrics = config.paths.artifacts / "metrics"
    detection = _load(metrics / "pallet_detection.json")
    loco = _load(config.paths.data / "manifests/loco.json")
    if not detection or detection.get("dataset") != "loco" or not loco:
        return None
    calibration = _load(metrics / "pallet_count_calibration.json")
    count = (
        f"{calibration['test_mae']:.1f} / {calibration['test_bias']:+.1f}"
        if calibration
        else f"{detection['pallet_count_mae']:.1f} / –"
    )
    train = loco["splits"]["train"]
    return (
        f"| {label} | sessions {_sessions(train['subsets'])}, {train['images']:,} images; "
        f"{training} | {_fmt(detection['pallet_AP50'])} | {_fmt(detection['pallet_AP50_95'])} | "
        f"{_fmt(detection['mAP50'])} | {count} |"
    )


def detection_table(config: Config, final: Config | None) -> list[str]:
    rows = [
        "| YOLO11n, 640 px | Trained on (LOCO) | Pallet AP50 | Pallet AP50-95 | "
        "mAP50, 5 classes | Pallet count MAE / bias |",
        "|---|---|---|---|---|---|",
    ]
    candidates = []
    if final is not None:
        candidates.append(
            ("**Final**", final, f"{final.vision.detector.epochs} epochs, last checkpoint")
        )
    candidates.append(
        (
            "Development",
            config,
            "epoch selected on session " + _sessions(config.vision.loco.val_subsets),
        )
    )
    for label, cfg, training in candidates:
        row = _detector_row(label, cfg, training)
        if row:
            rows.append(row)
    if len(rows) == 2:
        rows.append("| YOLO11n | LOCO | not run | | | |")
    loco = _load(config.paths.data / "manifests/loco.json")
    if loco:
        test = loco["splits"]["test"]
        rows += [
            "",
            f"Both detectors are scored on LOCO sessions {_sessions(test['subsets'])} "
            f"({test['images']:,} images), which were never used for training or model selection. "
            "Pallet counts use confidence 0.25, selected on validation data.",
        ]
    return rows


def results_table(config: Config, final: Config | None = None) -> str:
    metrics = config.paths.artifacts / "metrics"
    rows = detection_table(config, final)
    rows += ["", "| LayoutLMv3-base | Trained on | Evaluated on | Entity F1 |", "|---|---|---|---|"]
    for suffix, label in [("", "word boxes (OCR setting)"), ("-segment", "segment boxes")]:
        funsd = _load(metrics / f"document_layoutlm_funsd{suffix}.json")
        split = funsd["split"] if funsd else {"train": "-", "test": "-"}
        rows.append(
            f"| {label} | FUNSD, {split['train']} scanned forms | "
            f"FUNSD official test, {split['test']} forms | "
            f"{_fmt(funsd['f1'] if funsd else None)} |"
        )
    latency = _load(metrics / "latency.json")
    if latency:
        devices = sorted({d for key in LATENCY_MODELS for d in latency.get(key, {})})
        rows += [
            "",
            f"Median single-image latency on {latency['hardware']} (measured with the development "
            "checkpoint; the final model has the same architecture and input size):",
            "",
            "| Model | " + " | ".join(d.upper() for d in devices) + " |",
            "|---|" + "---|" * len(devices),
        ]
        for key, label in LATENCY_MODELS.items():
            values = latency.get(key, {})
            cells = [f"{values[d]['median_ms']:.0f} ms" if d in values else "-" for d in devices]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def update_readme(
    config: Config, readme: Path = Path("README.md"), final_config: Path | None = FINAL_CONFIG
) -> dict:
    final = load_config(final_config) if final_config and final_config.exists() else None
    table = results_table(config, final)
    source = readme.read_text(encoding="utf-8")
    if START not in source or END not in source:
        raise ValueError(f"{readme} has no real-results markers")
    head, rest = source.split(START, 1)
    readme.write_text(head + START + "\n" + table + "\n" + END + rest.split(END, 1)[1])
    return {"readme": str(readme), "rows": table.count("\n") - 1}
