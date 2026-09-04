"""Orchestration over reusable inference, tracking and reconciliation modules."""

import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from freightlens.config import Config
from freightlens.data.validation import bootstrap
from freightlens.document_ai.generator import generate_documents, generate_one
from freightlens.document_ai.layoutlm import train_layoutlm
from freightlens.document_ai.metrics import evaluate_documents
from freightlens.document_ai.ocr import FixtureOCRBackend, PaddleOCRBackend
from freightlens.document_ai.parser import parse_document
from freightlens.document_ai.preprocess import load_pages
from freightlens.io import read_json, record_run, write_json
from freightlens.reconciliation.metrics import run_reconciliation
from freightlens.reconciliation.rules import reconcile
from freightlens.reconciliation.scenarios import generate_scenarios
from freightlens.reconciliation.state import expected_from_document, observed_from_tracking
from freightlens.vision.tracker import track_video
from freightlens.vision.tracking_metrics import run_tracking
from freightlens.vision.video_generator import generate_videos


def analyze_document(path: Path, config: Config, fixture: Path | None = None):
    if fixture is not None:
        if config.document.ocr.backend != "fixture":
            raise ValueError("Fixture input requires explicitly configured fixture backend")
        ocr = FixtureOCRBackend(fixture).extract(load_pages(path)[0])
    else:
        if config.document.ocr.backend == "fixture":
            raise ValueError("Fixture mode cannot OCR arbitrary uploads. Configure PaddleOCR.")
        pages = load_pages(path)
        engine = PaddleOCRBackend()
        ocr = engine.extract(pages[0])
        for index, image in enumerate(pages[1:], 1):
            extra = engine.extract(image)
            for token in extra.tokens:
                token.page = index
            ocr.tokens.extend(extra.tokens)
    return parse_document(ocr, path.stem), ocr


def end_to_end(config: Config) -> dict:
    directory = config.paths.artifacts / "demo"
    annotation = generate_one(directory, 900001, config.seed, "demo", pallet_count=3)
    path = annotation.with_suffix(".png")
    document, ocr = analyze_document(path, config, annotation if config.mode == "smoke" else None)
    video = config.paths.data / "synthetic/tracking/VIDEO_000.mp4"
    truth = read_json(video.with_suffix(".json"))
    tracking = track_video(
        video,
        config,
        line=truth["line"],
        zone=truth["zone"],
        output=directory / "tracking_demo.mp4",
    )
    expected = expected_from_document(document)
    observed = observed_from_tracking(tracking, expected.shipment_id)
    result = reconcile(expected, observed, config.reconciliation)
    report = {
        "document": document.model_dump(),
        "expected": expected.model_dump(),
        "observed": observed.model_dump(),
        "verification": result.model_dump(),
        "provenance": {
            "ocr_backend": ocr.backend,
            "tracking_backend": tracking["backend"],
            "fixture": config.mode == "smoke",
            "vehicle_id_source": "unobserved",
        },
    }
    write_json(directory / "report.json", report)
    with Image.open(path) as image:
        annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for token in ocr.tokens:
        draw.rectangle(token.box, outline="#07A987", width=1)
    annotated.save(directory / "document_annotated.png")
    shutil.copyfile(
        directory / "tracking_demo.mp4", config.paths.artifacts / "videos/tracking_demo.mp4"
    )
    return report


def generate_report(config: Config, update_docs: bool = False) -> dict:
    metrics_dir = config.paths.artifacts / "metrics"
    names = [
        "document_baseline",
        "document_layoutlm",
        "pallet_detection",
        "tracking",
        "reconciliation",
    ]
    summary: dict[str, Any] = {}
    for name in names:
        path = metrics_dir / f"{name}.json"
        summary[name] = read_json(path) if path.exists() else {"status": "not_run"}
    for name in [
        "document_paddleocr",
        "document_layoutlm_model_probe",
        "pallet_detection_model_probe",
    ]:
        path = metrics_dir / f"{name}.json"
        if path.exists():
            summary[name] = read_json(path)
    write_json(metrics_dir / "summary.json", summary)
    rows = ["| Component | Evaluation | Metric | Value |", "|---|---|---|---|"]
    measured_rows = [
        ("document_baseline", "f1"),
        ("document_layoutlm", "f1"),
        ("pallet_detection", "mAP50"),
        ("tracking", "crossing_f1"),
        ("reconciliation", "f1"),
    ]
    if "document_paddleocr" in summary:
        measured_rows.append(("document_paddleocr", "f1"))
    if (
        summary["document_layoutlm"].get("status") != "measured"
        and "document_layoutlm_model_probe" in summary
    ):
        measured_rows.append(("document_layoutlm_model_probe", "f1"))
    for name, metric in measured_rows:
        entry = summary[name]
        value = entry.get(metric)
        formatted = f"{value:.4f}" if value is not None else "not run"
        scope = entry.get("evaluation_scope", entry["status"])
        rows.append(f"| {name} | {scope} | {metric} | {formatted} |")
    table = "\n".join(rows)
    (metrics_dir / "results.md").write_text(table + "\n", encoding="utf-8")
    if "breakdown" in summary["document_baseline"]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        breakdown = summary["document_baseline"]["breakdown"]
        levels = [
            level
            for level in ["clean", "light", "medium", "hard"]
            if f"degradation/{level}" in breakdown
        ]
        figure, axis = plt.subplots(figsize=(6, 3.5))
        axis.bar(
            levels, [breakdown[f"degradation/{level}"]["f1"] for level in levels], color="#167D8D"
        )
        axis.set(ylim=(0, 1.05), ylabel="Field F1", title="Document extraction by degradation")
        figure.text(0.1, 0.01, summary["document_baseline"]["evaluation_scope"], fontsize=8)
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        destination = config.paths.artifacts / "figures/document_f1.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
        plt.close(figure)
    docs = [Path("README.md"), Path("REPORT.md")] if update_docs else []
    for path in docs:
        if path.exists():
            source = path.read_text(encoding="utf-8")
            start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
            if start in source and end in source:
                source = (
                    source.split(start)[0]
                    + start
                    + "\n"
                    + table
                    + "\n"
                    + end
                    + source.split(end)[1]
                )
                path.write_text(source, encoding="utf-8")
    return summary


def run_demo(config: Config) -> dict:
    bootstrap(config)
    generate_documents(config)
    bootstrap(config)
    evaluate_documents(config)
    train_layoutlm(config)
    if config.mode == "full":
        from freightlens.vision.detector import train_detector
        from freightlens.vision.palleset import download_subset, prepare_palleset

        download_subset(config)
        prepare_palleset(config)
        train_detector(config)
    else:
        path = config.paths.artifacts / "metrics/pallet_detection.json"
        if not path.exists():
            write_json(
                path,
                {
                    "status": "not_run",
                    "mAP50": None,
                    "reason": "Offline smoke does not download detector weights or data",
                },
            )
    generate_videos(config)
    run_tracking(config)
    generate_scenarios(config)
    run_reconciliation(config)
    report = end_to_end(config)
    generate_report(config)
    record_run(config, "demo", report)
    return report
