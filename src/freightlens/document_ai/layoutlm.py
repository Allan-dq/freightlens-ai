"""Frozen-backbone LayoutLMv3 token classifier with train/val/test isolation."""

from pathlib import Path

import numpy as np
from PIL import Image

from freightlens.config import Config
from freightlens.data.seeds import set_global_seed
from freightlens.document_ai.generator import FIELDS
from freightlens.io import read_json, record_run, write_json

LABELS = ["O"] + [f"{prefix}-{field.upper()}" for field in FIELDS for prefix in ["B", "I"]]


def prepare_sample(path: Path) -> dict:
    annotation = read_json(path)
    width, height = annotation["image_size"]
    boxes = [
        [
            int(np.clip(v / (width if i % 2 == 0 else height) * 1000, 0, 1000))
            for i, v in enumerate(token["box"])
        ]
        for token in annotation["tokens"]
    ]
    for box in boxes:
        if box[0] > box[2] or box[1] > box[3]:
            raise ValueError("Invalid normalized box")
    return {
        "image": str(path.with_suffix(".png")),
        "words": [t["text"] for t in annotation["tokens"]],
        "boxes": boxes,
        "labels": [LABELS.index(t["label"]) for t in annotation["tokens"]],
    }


def prepare_documents(config: Config) -> dict:
    result = {}
    for split in ["train", "val", "test"]:
        samples = [
            prepare_sample(path)
            for path in sorted(
                (config.paths.data / "synthetic/documents" / split).glob("DOC_*.json")
            )
        ]
        if not samples:
            raise ValueError(f"No {split} documents; generate documents first")
        write_json(config.paths.data / f"processed/documents/{split}.json", samples)
        result[split] = len(samples)
    return result


def _dependencies():
    try:
        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    except ImportError as error:
        raise RuntimeError("Install LayoutLMv3 dependencies: uv sync --extra layout") from error
    return torch, LayoutLMv3ForTokenClassification, LayoutLMv3Processor


def _encode(processor, samples):
    images = []
    for sample in samples:
        with Image.open(sample["image"]) as image:
            images.append(image.convert("RGB"))
    return processor(
        images,
        [s["words"] for s in samples],
        boxes=[s["boxes"] for s in samples],
        word_labels=[s["labels"] for s in samples],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )


def _evaluate(model, processor, samples, batch_size):
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    torch, _, _ = _dependencies()
    predicted, expected, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _encode(processor, samples[start : start + batch_size])
            output = model(**batch)
            losses.append(float(output.loss))
            labels = batch["labels"].numpy()
            guesses = output.logits.argmax(-1).numpy()
            for truth, guess in zip(labels, guesses, strict=True):
                mask = truth != -100
                expected.append([LABELS[int(v)] for v in truth[mask]])
                predicted.append([LABELS[int(v)] for v in guess[mask]])
    return {
        "loss": float(np.mean(losses)),
        "precision": float(precision_score(expected, predicted)),
        "recall": float(recall_score(expected, predicted)),
        "f1": float(f1_score(expected, predicted)),
        "per_entity": classification_report(expected, predicted, output_dict=True, zero_division=0),
        "document_token_exact_match": float(
            np.mean([a == b for a, b in zip(expected, predicted, strict=True)])
        ),
        "samples": len(samples),
    }


def train_layoutlm(config: Config, force: bool = False) -> dict:
    counts = prepare_documents(config)
    target = config.paths.artifacts / "metrics/document_layoutlm.json"
    if not config.document.layoutlm.enabled and not force:
        result = {
            "status": "not_run",
            "reason": "Offline smoke validates labeled inputs only; "
            "use --enable-model for pretrained processor + inference + optimizer step.",
            "prepared_samples": counts,
            "f1": None,
        }
        write_json(target, result)
        return result
    torch, model_class, processor_class = _dependencies()
    torch.set_num_threads(2)
    set_global_seed(config.seed)
    settings = config.document.layoutlm
    processor = processor_class.from_pretrained(settings.model_name, apply_ocr=False)
    model = model_class.from_pretrained(
        settings.model_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    if settings.freeze_backbone:
        for parameter in model.layoutlmv3.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    root = config.paths.data / "processed/documents"
    train, val = read_json(root / "train.json"), read_json(root / "val.json")
    if config.mode == "smoke":
        train, val = train[:2], val[:2]
    best, stale, history = -1.0, 0, []
    destination = config.paths.models / "document/layoutlmv3"
    for epoch in range(settings.epochs):
        order = np.random.default_rng(config.seed + epoch).permutation(len(train))
        model.train()
        if settings.freeze_backbone:
            model.layoutlmv3.eval()
        for start in range(0, len(order), settings.batch_size):
            batch = _encode(
                processor, [train[i] for i in order[start : start + settings.batch_size]]
            )
            optimizer.zero_grad()
            model(**batch).loss.backward()
            optimizer.step()
        validation = _evaluate(model, processor, val, settings.batch_size)
        history.append({"epoch": epoch + 1, **validation})
        if validation["f1"] > best:
            best, stale = validation["f1"], 0
            model.save_pretrained(destination)
            processor.save_pretrained(destination)
        else:
            stale += 1
            if stale >= settings.patience:
                break
    write_json(destination / "training_history.json", history)
    return evaluate_layoutlm(config)


def evaluate_layoutlm(config: Config) -> dict:
    _, model_class, processor_class = _dependencies()
    checkpoint = config.paths.models / "document/layoutlmv3"
    if not checkpoint.exists():
        raise ValueError("Train LayoutLMv3 before evaluating")
    samples = read_json(config.paths.data / "processed/documents/test.json")
    if config.mode == "smoke":
        samples = samples[:2]
    metrics = _evaluate(
        model_class.from_pretrained(checkpoint),
        processor_class.from_pretrained(checkpoint, apply_ocr=False),
        samples,
        config.document.layoutlm.batch_size,
    )
    metrics.update(
        status="measured",
        evaluation_scope="ground_truth_tokens_layout_entity_f1",
        checkpoint=str(checkpoint),
        pretrained_source=config.document.layoutlm.model_name,
    )
    write_json(config.paths.artifacts / "metrics/document_layoutlm.json", metrics)
    write_json(config.paths.artifacts / "metrics/document_layoutlm_model_probe.json", metrics)
    record_run(config, "layoutlm", metrics)
    return metrics
