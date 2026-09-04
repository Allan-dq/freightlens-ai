from pathlib import Path

from PIL import Image

from freightlens.config import Config
from freightlens.data.registry import DATASETS
from freightlens.io import read_json, sha256, write_json
from freightlens.schemas.document import ShipmentDocument


def validate_box(box, width: int, height: int) -> None:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Out-of-bounds or empty box {box} in {width}x{height}")


def validate_documents(paths: list[Path]) -> dict:
    seen = set()
    for path in paths:
        item = read_json(path)
        if item["id"] in seen:
            raise ValueError(f"Duplicate document ID: {item['id']}")
        seen.add(item["id"])
        with Image.open(path.with_suffix(".png")) as image:
            image.load()
            width, height = image.size
        if not path.with_suffix(".pdf").is_file():
            raise ValueError(f"Missing PDF: {path}")
        for token in item["tokens"]:
            validate_box(token["box"], width, height)
        ShipmentDocument(document_id=item["id"], document_type=item["document_type"],
                         **item["fields"])
    return {"valid": True, "documents": len(seen)}


def bootstrap(config: Config) -> dict:
    for path in [config.paths.data, config.paths.models, config.paths.artifacts]:
        path.mkdir(parents=True, exist_ok=True)
    paths = sorted((config.paths.data / "synthetic/documents").glob("*/DOC_*.json"))
    report = validate_documents(paths)
    manifest_path = config.paths.data / "manifests/documents.json"
    if manifest_path.exists() and paths:
        for sample in read_json(manifest_path)["samples"]:
            if sha256(Path(sample["path"])) != sample["sha256"]:
                raise ValueError(f"Manifest hash mismatch: {sample['path']}")
    report.update(registry=DATASETS, seed=config.seed,
                  status="validated" if paths else "initialized_no_samples_yet")
    write_json(config.paths.artifacts / "metrics/data_validation.json", report)
    return report
