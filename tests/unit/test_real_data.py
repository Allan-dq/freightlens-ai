from pathlib import Path

import pytest

from freightlens.config import load_config
from freightlens.devices import resolve_device
from freightlens.document_ai.funsd_layoutlm import FUNSD_LABELS, form_to_sample
from freightlens.vision.loco import class_names, coco_to_yolo, subset_labels


def test_coco_box_converts_to_normalised_yolo():
    assert coco_to_yolo([10, 20, 30, 40], 100, 200) == pytest.approx([0.25, 0.2, 0.3, 0.2])


def test_coco_box_is_clipped_and_degenerate_boxes_dropped():
    assert coco_to_yolo([-10, -10, 30, 30], 100, 100) == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert coco_to_yolo([99.5, 10, 5, 5], 100, 100) is None


def test_loco_labels_use_contiguous_class_order():
    coco = {
        "categories": [{"id": 7, "name": "Forklift"}, {"id": 3, "name": "pallet"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 50, 50]},
            {"image_id": 1, "category_id": 7, "bbox": [50, 50, 50, 50], "iscrowd": 1},
        ],
    }
    names = class_names(coco["categories"])
    assert names == ["pallet", "forklift"]
    rows = subset_labels(coco, names)
    assert rows == {"a.jpg": ["0 0.250000 0.250000 0.500000 0.500000"]}


def test_funsd_entities_become_bio_tags_with_1000_grid_boxes():
    annotation = {
        "form": [
            {
                "label": "question",
                "words": [
                    {"text": "Date", "box": [0, 0, 50, 10]},
                    {"text": ":", "box": [50, 0, 55, 10]},
                ],
            },
            {"label": "answer", "words": [{"text": "1998", "box": [60, 0, 100, 10]}]},
            {
                "label": "other",
                "words": [
                    {"text": " ", "box": [0, 0, 1, 1]},
                    {"text": "Page", "box": [0, 90, 20, 100]},
                ],
            },
        ]
    }
    sample = form_to_sample(annotation, Path("f.png"), width=100, height=100)
    assert sample["words"] == ["Date", ":", "1998", "Page"]
    assert [FUNSD_LABELS[i] for i in sample["labels"]] == [
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "O",
    ]
    assert sample["boxes"][2] == [600, 0, 1000, 100]


def test_real_config_selects_loco_and_auto_device():
    config = load_config("configs/real.yaml")
    assert config.vision.detector.dataset == "loco"
    assert config.vision.detector.device == "auto"
    assert not set(config.vision.loco.train_subsets) & set(config.vision.loco.test_subsets)
    assert resolve_device("cpu") == "cpu"


def test_funsd_segment_boxes_share_the_entity_box():
    annotation = {
        "form": [
            {
                "label": "question",
                "box": [0, 0, 50, 10],
                "words": [
                    {"text": "Ship", "box": [0, 0, 20, 10]},
                    {"text": "to", "box": [25, 0, 50, 10]},
                ],
            }
        ]
    }
    word = form_to_sample(annotation, Path("f.png"), 100, 100, box_level="word")
    segment = form_to_sample(annotation, Path("f.png"), 100, 100, box_level="segment")
    assert word["boxes"] == [[0, 0, 200, 100], [250, 0, 500, 100]]
    assert segment["boxes"] == [[0, 0, 500, 100], [0, 0, 500, 100]]
    with pytest.raises(ValueError):
        form_to_sample(annotation, Path("f.png"), 100, 100, box_level="line")


def test_real_results_table_renders_measured_values_only(tmp_path):
    from freightlens.io import write_json
    from freightlens.real_results import END, START, results_table, update_readme

    config = load_config("configs/real.yaml")
    config.paths.artifacts = tmp_path / "artifacts"
    config.paths.data = tmp_path / "data"
    write_json(
        config.paths.data / "manifests/loco.json",
        {"splits": {"train": {"images": 1943}, "test": {"images": 2277}}},
    )
    write_json(
        config.paths.artifacts / "metrics/pallet_detection.json",
        {
            "dataset": "loco",
            "pallet_AP50": 0.5,
            "pallet_AP50_95": 0.25,
            "mAP50": 0.4,
            "pallet_count_mae": 3.0,
        },
    )
    table = results_table(config)
    assert "| pallet count error per image (MAE) | 3.0 |" in table
    write_json(
        config.paths.artifacts / "metrics/pallet_count_calibration.json",
        {"threshold": 0.25, "test_mae": 7.21, "test_bias": -0.91},
    )
    table = results_table(config)
    assert "threshold 0.25 chosen on validation (MAE / bias) | 7.2 / -0.9 |" in table
    assert "| pallet AP50 | 0.500 |" in table
    assert "1,943 real warehouse images" in table
    assert table.count("not run") == 2

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nSTALE_ROW\n{END}\nfooter\n", encoding="utf-8")
    update_readme(config, readme)
    text = readme.read_text(encoding="utf-8")
    assert "STALE_ROW" not in text and text.endswith("footer\n") and "0.500" in text


def test_real_results_latency_is_a_separate_device_table(tmp_path):
    from freightlens.io import write_json
    from freightlens.real_results import results_table

    config = load_config("configs/real.yaml")
    config.paths.artifacts = tmp_path / "artifacts"
    config.paths.data = tmp_path / "data"
    write_json(
        config.paths.artifacts / "metrics/latency.json",
        {
            "hardware": "Test CPU",
            "detector_yolo11n_640": {"cpu": {"median_ms": 36.2}, "mps": {"median_ms": 18.6}},
            "layoutlmv3_base_512_tokens": {"cpu": {"median_ms": 263.0}},
        },
    )
    table = results_table(config)
    assert "| Model | CPU | MPS |" in table
    assert "| YOLO11n, 640 px input | 36 ms | 19 ms |" in table
    assert "| LayoutLMv3-base, 512 tokens (OCR excluded) | 263 ms | - |" in table


def test_count_threshold_is_selected_by_validation_mae():
    import numpy as np

    from freightlens.vision.calibration import count_errors, select_threshold

    confidences = [np.array([0.9, 0.6, 0.3]), np.array([0.8, 0.2]), np.array([0.7, 0.5, 0.4])]
    truths = np.array([2, 1, 3])
    assert count_errors(confidences, truths, 0.5) == (1 / 3, -1 / 3)
    choice = select_threshold(confidences, truths, grid=[0.25, 0.35, 0.45, 0.65])
    assert choice["threshold"] == 0.35
    assert choice["mae"] == 0.0
    assert [row["threshold"] for row in choice["curve"]] == [0.25, 0.35, 0.45, 0.65]
