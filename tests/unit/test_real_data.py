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
