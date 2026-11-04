"""Compare full-frame and tiled inference on LOCO.

Run on the validation split first; it selects a counting threshold per mode and saves the
report. Then run on the test split, which reuses those thresholds and is scored once.

    python scripts/evaluate_tiling.py --split val
    python scripts/evaluate_tiling.py --split test --modes full_frame_640 tiles_640_plus_full
"""

import argparse
import json

from freightlens.config import load_config
from freightlens.vision.tiling import DEFAULT_MODES, evaluate_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--modes", nargs="+", default=[m.name for m in DEFAULT_MODES])
    args = parser.parse_args()
    config = load_config(args.config)
    modes = [m for m in DEFAULT_MODES if m.name in args.modes]
    thresholds = None
    if args.split == "test":
        validation = json.loads(
            (config.paths.artifacts / "metrics/tiling_val.json").read_text(encoding="utf-8")
        )
        thresholds = {m.name: validation["modes"][m.name]["count_threshold"] for m in modes}
    summary = evaluate_split(config, args.split, modes, thresholds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
