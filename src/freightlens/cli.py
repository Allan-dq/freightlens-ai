"""CLI and script entrypoints use the same dispatcher."""

import argparse
import json
from importlib import import_module
from pathlib import Path

from freightlens import __version__
from freightlens.config import load_config
from freightlens.io import read_json
from freightlens.logging import configure_logging

COMMANDS = {
    "bootstrap": ("data.validation", "bootstrap"),
    "generate-documents": ("document_ai.generator", "generate_documents"),
    "download-palleset": ("vision.palleset", "download_subset"),
    "download-funsd": ("data.download", "download_funsd"),
    "prepare-documents": ("document_ai.layoutlm", "prepare_documents"),
    "prepare-palleset": ("vision.palleset", "prepare_palleset"),
    "train-detector": ("vision.detector", "train_detector"),
    "evaluate-detector": ("vision.metrics", "evaluate_detector"),
    "generate-tracking-videos": ("vision.video_generator", "generate_videos"),
    "run-tracking": ("vision.tracking_metrics", "run_tracking"),
    "generate-scenarios": ("reconciliation.scenarios", "generate_scenarios"),
    "run-reconciliation": ("reconciliation.metrics", "run_reconciliation"),
    "evaluate-reconciliation": ("reconciliation.metrics", "run_reconciliation"),
    "demo": ("pipeline", "run_demo"),
    "run-end-to-end-demo": ("pipeline", "run_demo"),
    "report": ("pipeline", "generate_report"),
}


def dispatch(action: str, config, args):
    if action == "report":
        from freightlens.pipeline import generate_report

        return generate_report(config, update_docs=True)
    if action == "download-palleset":
        for split in ["train", "val", "test"]:
            value = getattr(args, f"{split}_size", None)
            if value is not None:
                setattr(config.vision.palleset, f"{split}_size", value)
    if action in COMMANDS:
        module, function = COMMANDS[action]
        return getattr(import_module(f"freightlens.{module}"), function)(config)
    if action == "train-layoutlm":
        from freightlens.document_ai.layoutlm import train_layoutlm

        return train_layoutlm(config, force=args.enable_model)
    if action == "evaluate-documents":
        from freightlens.document_ai.metrics import evaluate_documents

        return evaluate_documents(config, backend=args.backend)
    if action == "analyze-document":
        from freightlens.pipeline import analyze_document

        result, ocr = analyze_document(
            Path(args.path), config, Path(args.fixture) if args.fixture else None
        )
        return {
            "document": result.model_dump(),
            "ocr_backend": ocr.backend,
            "fixture": ocr.is_fixture,
        }
    if action in ["detect", "track"]:
        from freightlens.vision.detector import detect
        from freightlens.vision.tracker import track_video

        return (detect if action == "detect" else track_video)(Path(args.path), config)
    if action == "reconcile":
        from freightlens.reconciliation.rules import reconcile
        from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState

        return reconcile(
            ExpectedShipmentState.model_validate(read_json(Path(args.expected))),
            ObservedShipmentState.model_validate(read_json(Path(args.observed))),
            config.reconciliation,
        ).model_dump()
    raise ValueError(f"Unknown command: {action}")


def _options(parser):
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--backend", choices=["rules", "layoutlm"], default="rules")
    parser.add_argument("--enable-model", action="store_true")
    parser.add_argument("--fixture", help="Generated OCR annotation for explicit offline testing")
    for split in ["train", "val", "test"]:
        parser.add_argument(f"--{split}-size", type=int)


def script_main(action: str) -> None:
    parser = argparse.ArgumentParser(description=f"FreightLens {action}")
    _options(parser)
    args = parser.parse_args()
    configure_logging()
    result = dispatch(action, load_config(args.config), args)
    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="freightlens", description="Freight verification pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    mapping = {
        "data": {
            "generate-documents": "generate-documents",
            "download-palleset": "download-palleset",
            "download-funsd": "download-funsd",
            "prepare-palleset": "prepare-palleset",
        },
        "document": {
            "analyze": "analyze-document",
            "evaluate": "evaluate-documents",
            "train": "train-layoutlm",
        },
        "vision": {
            "detect": "detect",
            "track": "track",
            "evaluate": "evaluate-detector",
            "train": "train-detector",
        },
    }
    for group, commands in mapping.items():
        group_parser = sub.add_parser(group)
        children = group_parser.add_subparsers(dest="subcommand", required=True)
        for name, action in commands.items():
            child = children.add_parser(name)
            _options(child)
            child.set_defaults(action=action)
            if name in ["analyze", "detect", "track"]:
                child.add_argument("path")
    for name in ["demo", "report", "reconcile", "bootstrap"]:
        child = sub.add_parser(name)
        _options(child)
        child.set_defaults(action=name)
        if name == "reconcile":
            child.add_argument("--expected", required=True)
            child.add_argument("--observed", required=True)
    args = parser.parse_args()
    if args.command == "version":
        print(__version__)
        return
    configure_logging()
    try:
        result = dispatch(args.action, load_config(args.config), args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"freightlens: {error}\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
