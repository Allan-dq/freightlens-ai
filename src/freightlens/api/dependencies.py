import os
from functools import lru_cache

from freightlens.config import Config, load_config


@lru_cache
def get_config() -> Config:
    config = load_config(os.getenv("FREIGHTLENS_CONFIG", "configs/base.yaml"))
    if "FREIGHTLENS_MAX_UPLOAD_MB" in os.environ:
        config.api.max_upload_mb = int(os.environ["FREIGHTLENS_MAX_UPLOAD_MB"])
        if config.api.max_upload_mb <= 0:
            raise ValueError("Upload limit must be positive")
    return config
