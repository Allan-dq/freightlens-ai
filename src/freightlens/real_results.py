"""Render the real-data results table from saved metric JSON.

Only measured values are rendered; a stage without metrics is listed as "not run".
The table replaces the block between REAL_RESULTS markers in README.md.
"""

from pathlib import Path

from freightlens.config import Config
from freightlens.io import read_json

START, END = "<!-- REAL_RESULTS_START -->", "<!-- REAL_RESULTS_END -->"


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
        model = "YOLO11n detector"
        rows += [
            f"| {model} | {trained} | {tested} | pallet AP50 | {_fmt(detection['pallet_AP50'])} |",
            f"| {model} | | | pallet AP50-95 | {_fmt(detection['pallet_AP50_95'])} |",
            f"| {model} | | | mAP50, all 5 classes | {_fmt(detection['mAP50'])} |",
            f"| {model} | | | pallet count MAE per image | "
            f"{_fmt(detection['pallet_count_mae'], 2)} |",
        ]
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
        for key, label in [
            ("detector_yolo11n_640", "YOLO11n, 640 px"),
            ("layoutlmv3_base_512_tokens", "LayoutLMv3-base, 512 tokens"),
        ]:
            for device, values in latency[key].items():
                rows.append(
                    f"| {label} | | {latency['hardware']}, {device} | median latency | "
                    f"{values['median_ms']:.0f} ms |"
                )
    return "\n".join(rows)


def update_readme(config: Config, readme: Path = Path("README.md")) -> dict:
    table = results_table(config)
    source = readme.read_text(encoding="utf-8")
    if START not in source or END not in source:
        raise ValueError(f"{readme} has no real-results markers")
    head, rest = source.split(START, 1)
    readme.write_text(head + START + "\n" + table + "\n" + END + rest.split(END, 1)[1])
    return {"readme": str(readme), "rows": table.count("\n") - 1}
