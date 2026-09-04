import os
from pathlib import Path

import pytest

from freightlens.config import load_config

settings_root = Path.cwd() / ".cache/ultralytics"
settings_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_root))


@pytest.fixture
def isolated_config(tmp_path):
    config = load_config("configs/smoke.yaml")
    config.paths.data = tmp_path / "data"
    config.paths.models = tmp_path / "models"
    config.paths.artifacts = tmp_path / "artifacts"
    config.document.synthetic.train_size = 4
    config.document.synthetic.val_size = 2
    config.document.synthetic.test_size = 4
    return config
