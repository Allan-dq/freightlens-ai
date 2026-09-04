import hashlib
from pathlib import Path

from freightlens.data.validation import bootstrap
from freightlens.io import sha256, write_json
from freightlens.vision.palleset import download_subset


def test_fresh_checkout_manifest_without_generated_data(isolated_config):
    write_json(
        isolated_config.paths.data / "manifests/documents.json",
        {"seed": 42, "samples": [{"path": "not-generated-yet.json", "sha256": "abc"}]},
    )
    assert bootstrap(isolated_config)["status"] == "initialized_no_samples_yet"


def test_missing_download_rehydrates_pinned_revision(isolated_config, monkeypatch):
    config = isolated_config
    config.vision.palleset.train_size = 1
    config.vision.palleset.val_size = 0
    config.vision.palleset.test_size = 0
    image, boxes = "train/rgb/rgb_0000.png", "train/bboxes_2d/box_0000.npy"
    manifest = {
        "seed": 42,
        "requested": {"train": 1, "val": 0, "test": 0},
        "revision": "fixed-commit",
        "pairs": {"train_000000": {"rgb": image, "bboxes_2d": boxes}},
        "train_ids": ["train_000000"],
        "val_ids": [],
        "test_ids": [],
        "download_bytes": 8,
        "files": [
            {"path": name, "sha256": hashlib.sha256(b"test").hexdigest(), "bytes": 4}
            for name in [image, boxes]
        ],
    }
    write_json(config.paths.data / "manifests/palleset_subset.json", manifest)
    calls = []

    def fetch(repo, name, repo_type, revision, local_dir, force_download):
        calls.append(revision)
        path = Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        return str(path)

    def forbid_listing():
        raise AssertionError("A saved experiment must not resolve today's main revision")

    monkeypatch.setattr("freightlens.vision.palleset.hf_hub_download", fetch)
    monkeypatch.setattr("freightlens.vision.palleset.HfApi", forbid_listing)
    result = download_subset(config)
    assert calls == ["fixed-commit", "fixed-commit"]
    assert result["revision"] == "fixed-commit"
    assert result["files"][0]["sha256"] == sha256(config.paths.data / "raw/palleset" / boxes)
