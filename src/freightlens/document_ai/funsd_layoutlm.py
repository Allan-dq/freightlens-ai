"""Full LayoutLMv3 fine-tuning on FUNSD, a public benchmark of real scanned forms.

FUNSD labels each semantic entity as header, question, answer or other. Words are
converted to BIO tags (other -> O) and boxes are normalised to the 0-1000 grid that
LayoutLMv3 expects. The official 149/50 train/test split is kept; a fixed slice of
the training forms is held out for checkpoint selection, so the test forms are only
used once, for the final score.

Two box conventions are supported. "word" gives every word its own box, which is
what an OCR engine produces at inference time. "segment" gives every word the box
of its annotated entity, the convention used for the published LayoutLMv3 FUNSD
result; it is useful for comparing against the paper but relies on ground-truth
entity extents that are not available on unseen documents.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from freightlens.config import Config
from freightlens.data.seeds import set_global_seed
from freightlens.devices import resolve_device
from freightlens.io import read_json, record_run, write_json

LOGGER = logging.getLogger(__name__)

FUNSD_LABELS = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]


def funsd_root(config: Config) -> Path:
    raw = config.paths.data / "raw/funsd"
    for candidate in sorted(raw.glob("**/training_data")):
        if "__MACOSX" not in candidate.parts:
            return candidate.parent
    raise ValueError("Download FUNSD first: freightlens data download-funsd")


def _normalise(box: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    scaled = [x0 / width, y0 / height, x1 / width, y1 / height]
    values = [int(np.clip(round(v * 1000), 0, 1000)) for v in scaled]
    values[2], values[3] = max(values[0], values[2]), max(values[1], values[3])
    return values


def form_to_sample(
    annotation: dict, image_path: Path, width: int, height: int, box_level: str = "word"
) -> dict:
    """Convert one FUNSD annotation into word-level LayoutLMv3 inputs with BIO tags."""
    if box_level not in {"word", "segment"}:
        raise ValueError(f"Unknown box level: {box_level}")
    words, boxes, labels = [], [], []
    for entity in annotation["form"]:
        kind = entity["label"].upper()
        entity_words = [w for w in entity["words"] if w["text"].strip()]
        for index, word in enumerate(entity_words):
            words.append(word["text"])
            source = entity["box"] if box_level == "segment" else word["box"]
            boxes.append(_normalise(source, width, height))
            if kind == "OTHER":
                labels.append(FUNSD_LABELS.index("O"))
            else:
                prefix = "B" if index == 0 else "I"
                labels.append(FUNSD_LABELS.index(f"{prefix}-{kind}"))
    return {
        "id": image_path.stem,
        "image": str(image_path),
        "words": words,
        "boxes": boxes,
        "labels": labels,
    }


def load_split(root: Path, split: str, box_level: str = "word") -> list[dict]:
    folder = root / ("training_data" if split == "train" else "testing_data")
    samples = []
    for path in sorted((folder / "annotations").glob("*.json")):
        if path.name.startswith("._"):
            continue
        image_path = folder / "images" / f"{path.stem}.png"
        with Image.open(image_path) as image:
            width, height = image.size
        samples.append(
            form_to_sample(read_json(path), image_path, width, height, box_level=box_level)
        )
    if not samples:
        raise ValueError(f"No FUNSD {split} forms under {folder}")
    return samples


def _dependencies():
    try:
        import torch
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError("Install LayoutLMv3 dependencies: uv sync --extra layout") from error
    return (
        torch,
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor,
        get_linear_schedule_with_warmup,
    )


def _encode(processor, samples, device):
    images = []
    for sample in samples:
        with Image.open(sample["image"]) as image:
            images.append(image.convert("RGB"))
    batch = processor(
        images,
        [s["words"] for s in samples],
        boxes=[s["boxes"] for s in samples],
        word_labels=[s["labels"] for s in samples],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    return batch.to(device)


def evaluate(model, processor, samples, batch_size: int, device: str) -> dict:
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    torch = _dependencies()[0]
    expected, predicted, words_scored, words_total = [], [], 0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = _encode(processor, chunk, device)
            logits = model(**batch).logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            for truth, guess in zip(labels, logits, strict=True):
                mask = truth != -100
                expected.append([FUNSD_LABELS[int(v)] for v in truth[mask]])
                predicted.append([FUNSD_LABELS[int(v)] for v in guess[mask]])
                words_scored += int(mask.sum())
            words_total += sum(len(s["words"]) for s in chunk)
    report = classification_report(expected, predicted, output_dict=True, zero_division=0)
    return {
        "precision": float(precision_score(expected, predicted)),
        "recall": float(recall_score(expected, predicted)),
        "f1": float(f1_score(expected, predicted)),
        "per_entity": {
            name: {k: float(v) for k, v in values.items()}
            for name, values in report.items()
            if name in ["HEADER", "QUESTION", "ANSWER"]
        },
        "forms": len(samples),
        "words_scored": words_scored,
        "words_total": words_total,
    }


def train_funsd_layoutlm(config: Config) -> dict:
    settings = config.document.funsd
    torch, model_class, processor_class, schedule = _dependencies()
    device = resolve_device(settings.device)
    set_global_seed(config.seed)
    root = funsd_root(config)
    level = settings.box_level
    train_all, test = load_split(root, "train", level), load_split(root, "test", level)
    order = np.random.default_rng(config.seed).permutation(len(train_all))
    val = [train_all[i] for i in order[: settings.val_forms]]
    train = [train_all[i] for i in order[settings.val_forms :]]
    if config.mode == "smoke":
        train, val, test = train[:2], val[:1], test[:1]

    processor = processor_class.from_pretrained(settings.model_name, apply_ocr=False)
    model = model_class.from_pretrained(
        settings.model_name,
        num_labels=len(FUNSD_LABELS),
        id2label=dict(enumerate(FUNSD_LABELS)),
        label2id={label: i for i, label in enumerate(FUNSD_LABELS)},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=0.01)
    max_steps = 2 if config.mode == "smoke" else settings.max_steps
    scheduler = schedule(optimizer, int(0.1 * max_steps), max_steps)
    suffix = "" if level == "word" else f"-{level}"
    destination = config.paths.models / f"document/layoutlmv3-funsd{suffix}"
    rng = np.random.default_rng(config.seed)
    best, history, step = -1.0, [], 0
    while step < max_steps:
        for start in range(0, len(train), settings.batch_size):
            if step >= max_steps:
                break
            if start == 0:
                rng.shuffle(train)
            model.train()
            batch = _encode(processor, train[start : start + settings.batch_size], device)
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % settings.eval_every == 0 or step == max_steps:
                validation = evaluate(model, processor, val, settings.batch_size, device)
                history.append({"step": step, "train_loss": loss.item(), **validation})
                LOGGER.info(
                    "step %d/%d train_loss=%.3f val_f1=%.4f",
                    step,
                    max_steps,
                    loss.item(),
                    validation["f1"],
                )
                if validation["f1"] > best:
                    best = validation["f1"]
                    model.save_pretrained(destination)
                    processor.save_pretrained(destination)
    write_json(destination / "training_history.json", history)

    best_model = model_class.from_pretrained(destination).to(device)
    metrics = evaluate(best_model, processor, test, settings.batch_size, device)
    metrics.update(
        status="measured",
        evaluation_scope="funsd_official_test_entity_f1",
        dataset="FUNSD (real scanned forms), official 149/50 split",
        split={"train": len(train), "val": len(val), "test": len(test)},
        pretrained_source=settings.model_name,
        fine_tuning="full",
        box_level=level,
        max_steps=max_steps,
        best_validation_f1=best,
        device=device,
        checkpoint=str(destination),
        note="Entity F1 (seqeval) over words with ground-truth text; "
        "words beyond 512 tokens are truncated and not scored.",
    )
    write_json(config.paths.artifacts / f"metrics/document_layoutlm_funsd{suffix}.json", metrics)
    record_run(config, f"layoutlm_funsd{suffix}", metrics)
    return metrics
