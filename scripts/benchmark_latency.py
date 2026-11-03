"""Measure end-to-end inference latency of the trained detector and form model.

Runs warm-up passes first, then times N single-image predictions on each device and
reports median and p95 in milliseconds together with the hardware description.
"""

import argparse
import platform
import statistics
import subprocess
import time

from freightlens.config import load_config
from freightlens.devices import resolve_device
from freightlens.io import write_json


def _hardware() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:
        return platform.processor()


def _summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "runs": len(ordered),
    }


def bench_detector(config, devices, images, runs):
    from freightlens.vision.detector import load_detector, trained_checkpoint

    model = load_detector(trained_checkpoint(config))
    size = config.vision.detector.image_size
    result = {}
    for device in devices:
        for path in images[:3]:
            model.predict(str(path), imgsz=size, device=device, verbose=False)
        timings = []
        for i in range(runs):
            start = time.perf_counter()
            model.predict(str(images[i % len(images)]), imgsz=size, device=device, verbose=False)
            timings.append((time.perf_counter() - start) * 1000)
        result[device] = _summary(timings)
    return result


def bench_layout(config, devices, samples, runs):
    import torch
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

    from freightlens.document_ai.funsd_layoutlm import _encode

    checkpoint = config.paths.models / "document/layoutlmv3-funsd"
    processor = LayoutLMv3Processor.from_pretrained(checkpoint, apply_ocr=False)
    result = {}
    for device in devices:
        model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint).to(device).eval()
        with torch.no_grad():
            for sample in samples[:3]:
                model(**_encode(processor, [sample], device))
            timings = []
            for i in range(runs):
                start = time.perf_counter()
                logits = model(**_encode(processor, [samples[i % len(samples)]], device)).logits
                logits.argmax(-1).cpu()
                timings.append((time.perf_counter() - start) * 1000)
        result[device] = _summary(timings)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    config = load_config(args.config)
    devices = sorted({"cpu", resolve_device("auto")})
    images = sorted((config.paths.data / "processed/loco/images/test").iterdir())[:50]
    from freightlens.document_ai.funsd_layoutlm import funsd_root, load_split

    forms = load_split(funsd_root(config), "test")
    result = {
        "hardware": _hardware(),
        "note": "Single image per call, after 3 warm-up calls. Detector time includes "
        "image decode and pre/post-processing; form time includes processor encoding "
        "with ground-truth words (OCR excluded).",
        "detector_yolo11n_640": bench_detector(config, devices, images, args.runs),
        "layoutlmv3_base_512_tokens": bench_layout(config, devices, forms, args.runs),
    }
    write_json(config.paths.artifacts / "metrics/latency.json", result)
    print(result)


if __name__ == "__main__":
    main()
