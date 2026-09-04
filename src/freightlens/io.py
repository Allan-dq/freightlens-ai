import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from freightlens.config import Config


def write_json(path: Path, value: Any) -> Path:
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=encode) + "\n",
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_run(config: Config, stage: str, metrics: dict) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = config.paths.artifacts / "runs" / f"{stamp}-{stage}-{uuid4().hex[:6]}"
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or None
        )
    except FileNotFoundError:
        commit = None
    write_json(output / "config.json", config.model_dump(mode="json"))
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "environment.json",
        {
            "timestamp": stamp,
            "seed": config.seed,
            "git_commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
        },
    )
    return output
