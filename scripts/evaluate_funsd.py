import argparse

from freightlens.config import load_config
from freightlens.document_ai.funsd import evaluate_funsd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    result = evaluate_funsd(load_config(args.config))
    print(f"Validated {result['samples']} external FUNSD forms")
