"""LOCO (Logistics Objects in Context) ingestion: real warehouse imagery, CC0.

LOCO was recorded by walking through real logistics facilities with low-cost
cameras. It ships COCO-format annotations split into five recording subsets.
Consecutive frames inside a subset are near-duplicates, so splits are made by
whole subset: training, validation and test never share a recording session.

Source: https://github.com/tum-fml/loco (Mayershofer et al., ICMLA 2020).
"""

import zipfile
from pathlib import Path

import yaml

from freightlens.config import Config
from freightlens.io import read_json, sha256, write_json

SOURCE = "https://github.com/tum-fml/loco"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def coco_to_yolo(bbox: list[float], width: int, height: int) -> list[float] | None:
    """Convert a COCO [x, y, w, h] box to normalised YOLO [cx, cy, w, h], clipped."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return [
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]


def class_names(categories: list[dict]) -> list[str]:
    """Contiguous class list ordered by COCO category id, lower-case with underscores."""
    ordered = sorted(categories, key=lambda c: c["id"])
    return [c["name"].strip().lower().replace(" ", "_") for c in ordered]


def subset_labels(coco: dict, names: list[str]) -> dict[str, list[str]]:
    """Map each image file name to its YOLO label rows."""
    order = {
        c["id"]: names.index(n)
        for c, n in zip(sorted(coco["categories"], key=lambda c: c["id"]), names, strict=True)
    }
    images = {image["id"]: image for image in coco["images"]}
    rows: dict[str, list[str]] = {image["file_name"]: [] for image in coco["images"]}
    for annotation in coco["annotations"]:
        if annotation.get("iscrowd"):
            continue
        image = images[annotation["image_id"]]
        box = coco_to_yolo(annotation["bbox"], image["width"], image["height"])
        if box is not None:
            rows[image["file_name"]].append(
                f"{order[annotation['category_id']]} " + " ".join(f"{v:.6f}" for v in box)
            )
    return rows


def _annotation_file(root: Path, subset: int) -> Path:
    matches = sorted(root.glob(f"loco-sub{subset}-v1-*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one annotation file for LOCO subset {subset} in {root}")
    return matches[0]


def _extract(root: Path) -> Path:
    images = root / "images"
    archive = root / "dataset.zip"
    if not images.exists():
        if not archive.exists():
            raise ValueError(f"Download LOCO dataset.zip into {root} first")
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                destination = (images / info.filename).resolve()
                if not destination.is_relative_to(images.resolve()):
                    raise ValueError("Unsafe path in LOCO archive")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(info))
    return images


def prepare_loco(config: Config) -> dict:
    root = config.paths.data / "raw/loco"
    target = config.paths.data / "processed/loco"
    image_root = _extract(root)
    by_name: dict[str, list[Path]] = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            by_name.setdefault(path.name, []).append(path)

    settings = config.vision.loco
    splits = {
        "train": settings.train_subsets,
        "val": settings.val_subsets,
        "test": settings.test_subsets,
    }
    flat = [s for subsets in splits.values() for s in subsets]
    if len(flat) != len(set(flat)):
        raise ValueError("A LOCO subset is assigned to more than one split")

    names: list[str] | None = None
    summary: dict = {"source": SOURCE, "license": "CC0-1.0", "splits": {}}
    for split, subsets in splits.items():
        images_dir, labels_dir = target / "images" / split, target / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {"images": 0, "boxes": 0, "missing_images": 0}
        for subset in subsets:
            coco = read_json(_annotation_file(root, subset))
            subset_names = class_names(coco["categories"])
            if names is None:
                names = subset_names
            elif subset_names != names:
                raise ValueError("LOCO subsets disagree on category names")
            for file_name, rows in subset_labels(coco, names).items():
                candidates = by_name.get(Path(file_name).name, [])
                if len(candidates) != 1:
                    counts["missing_images"] += 1
                    continue
                stem = f"s{subset}_{Path(file_name).stem}"
                link = images_dir / f"{stem}{candidates[0].suffix.lower()}"
                if not link.exists():
                    link.symlink_to(candidates[0].resolve())
                (labels_dir / f"{stem}.txt").write_text(
                    "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
                )
                counts["images"] += 1
                counts["boxes"] += len(rows)
        summary["splits"][split] = {**counts, "subsets": subsets}
    if names is None or "pallet" not in names:
        raise ValueError("LOCO annotations have no 'pallet' class")
    (target / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(target.resolve()),
                "train": "images/train",
                # Without a validation session (final fixed-epoch training) Ultralytics still
                # runs one end-of-training pass; point it at the training images, never test.
                "val": "images/val" if settings.val_subsets else "images/train",
                "test": "images/test",
                "names": dict(enumerate(names)),
            }
        ),
        encoding="utf-8",
    )
    summary["classes"] = names
    if (root / "dataset.zip").exists():
        summary["archive_sha256"] = sha256(root / "dataset.zip")
    write_json(config.paths.data / "manifests/loco.json", summary)
    return summary
