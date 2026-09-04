"""Selective, revision-pinned Palleset ingestion; never download other modalities."""

import random
import re
import shutil
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from freightlens.config import Config
from freightlens.data.validation import validate_box
from freightlens.io import read_json, sha256, write_json

REPO = "eusandre95/Palleset"


def identify_pairs(files: list[str]) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for name in sorted(files):
        match = re.search(
            r"(?:^|/)(train|val)/(rgb|bboxes_2d)/[^/]*?(\d+)\.(png|jpg|npy|json)$", name
        )
        if match:
            source, modality, number, _ = match.groups()
            key = f"{source}_{int(number):06d}"
            pairs.setdefault(key, {})[modality] = name
    return {key: value for key, value in pairs.items() if set(value) == {"rgb", "bboxes_2d"}}


def select_splits(pairs: dict, train_size: int, val_size: int, test_size: int, seed: int) -> dict:
    if min(train_size, val_size, test_size) < 0:
        raise ValueError("Subset sizes must be nonnegative")
    # Keep upstream train separate; subdivide upstream validation deterministically.
    train_ids = sorted(key for key in pairs if key.startswith("train_"))
    heldout = sorted(key for key in pairs if key.startswith("val_"))
    rng = random.Random(seed)
    rng.shuffle(train_ids)
    rng.shuffle(heldout)
    if train_size > len(train_ids) or val_size + test_size > len(heldout):
        raise ValueError("Requested subset exceeds available paired RGB/2D annotations")
    return {
        "train_ids": train_ids[:train_size],
        "val_ids": heldout[:val_size],
        "test_ids": heldout[val_size : val_size + test_size],
    }


def download_subset(config: Config) -> dict:
    root = config.paths.data / "raw/palleset"
    manifest_path = config.paths.data / "manifests/palleset_subset.json"
    sizes = config.vision.palleset
    requested = {"train": sizes.train_size, "val": sizes.val_size, "test": sizes.test_size}
    expected_hashes: dict[str, str] = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        expected_hashes = {file["path"]: file["sha256"] for file in manifest["files"]}
        if manifest["seed"] != config.seed or manifest["requested"] != requested:
            raise ValueError(
                "Existing Palleset manifest uses a different split. Use a different "
                "data root for another experiment; metrics must keep their split."
            )
        if all(
            (root / file["path"]).exists() and sha256(root / file["path"]) == file["sha256"]
            for file in manifest["files"]
        ):
            return manifest
        revision = manifest["revision"]
        pairs = manifest["pairs"]
        splits = {f"{split}_ids": manifest[f"{split}_ids"] for split in requested}
        names = sorted(file["path"] for file in manifest["files"])
        expected_bytes = manifest["download_bytes"]
    else:
        api = HfApi()
        revision = api.dataset_info(REPO).sha
        files = api.list_repo_files(REPO, repo_type="dataset", revision=revision)
        pairs = identify_pairs(files)
        splits = select_splits(
            pairs, sizes.train_size, sizes.val_size, sizes.test_size, config.seed
        )
        names = sorted(
            {
                pairs[key][modality]
                for ids in splits.values()
                for key in ids
                for modality in ["rgb", "bboxes_2d"]
            }
        )
        metadata = api.get_paths_info(REPO, names, repo_type="dataset", revision=revision)
        expected_bytes = sum(getattr(item, "size", 0) for item in metadata)
    if expected_bytes > sizes.max_download_mb * 1024 * 1024:
        raise ValueError(f"Subset requires {expected_bytes} bytes, exceeding download budget")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name in names:
        destination = root / name
        repair = (
            name in expected_hashes
            and destination.exists()
            and sha256(destination) != expected_hashes[name]
        )
        local = Path(
            hf_hub_download(
                REPO,
                name,
                repo_type="dataset",
                revision=revision,
                local_dir=root,
                force_download=repair,
            )
        )
        if name in expected_hashes and sha256(local) != expected_hashes[name]:
            raise ValueError(f"Downloaded file differs from saved manifest: {name}")
        records.append({"path": name, "sha256": sha256(local), "bytes": local.stat().st_size})
    manifest = {
        "repo_id": REPO,
        "revision": revision,
        "seed": config.seed,
        "requested": requested,
        **splits,
        "files": records,
        "pairs": {key: pairs[key] for ids in splits.values() for key in ids},
        "license": "CC-BY-4.0",
        "download_bytes": sum(r["bytes"] for r in records),
    }
    write_json(manifest_path, manifest)
    return manifest


def read_boxes(path: Path, width: int, height: int) -> list[list[float]]:
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.dtype.names:
            names = array.dtype.names
            required = ["x_min", "y_min", "x_max", "y_max"]
            if not all(name in names for name in required):
                raise ValueError(f"Unrecognized structured bbox fields: {names}")
            boxes = [[float(row[name]) for name in required] for row in array]
        else:
            raise ValueError("Unstructured annotation format requires an explicit converter")
    else:
        value = read_json(path)
        boxes = value["boxes"] if isinstance(value, dict) else value
    valid = []
    for box in boxes:
        # IsaacSim tight boxes can extend beyond the camera; clamp visible area explicitly.
        clipped = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        validate_box(clipped, width, height)
        valid.append(clipped)
    return valid


def to_yolo(box, width: int, height: int) -> list[float]:
    validate_box(box, width, height)
    x1, y1, x2, y2 = box
    return [
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def prepare_palleset(config: Config) -> Path:
    manifest = read_json(config.paths.data / "manifests/palleset_subset.json")
    source = config.paths.data / "raw/palleset"
    target = config.paths.data / "processed/palleset"
    seen = set()
    for split in ["train", "val", "test"]:
        for key in manifest[f"{split}_ids"]:
            if key in seen:
                raise ValueError(f"Split leakage: {key}")
            seen.add(key)
            pair = manifest["pairs"][key]
            with Image.open(source / pair["rgb"]) as image:
                image.load()
                width, height = image.size
            boxes = read_boxes(source / pair["bboxes_2d"], width, height)
            images, labels = target / "images" / split, target / "labels" / split
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / pair["rgb"], images / f"{key}.png")
            (labels / f"{key}.txt").write_text(
                "\n".join(
                    "0 " + " ".join(f"{v:.8f}" for v in to_yolo(box, width, height))
                    for box in boxes
                )
                + "\n",
                encoding="utf-8",
            )
    path = target / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(target.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "pallet"},
            }
        ),
        encoding="utf-8",
    )
    write_json(
        config.paths.artifacts / "metrics/palleset_validation.json",
        {"valid": True, "samples": len(seen), "revision": manifest["revision"]},
    )
    return path
