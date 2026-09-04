.PHONY: install lint test smoke data document vision reconcile demo report api
install:
	uv sync --extra dev
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/freightlens
test:
	uv run pytest -q
smoke:
	uv run freightlens demo --config configs/smoke.yaml
	uv run pytest -q
data:
	uv run freightlens data generate-documents --config configs/base.yaml
	uv run freightlens data download-palleset --config configs/base.yaml
	uv run freightlens data prepare-palleset --config configs/base.yaml
document:
	uv run --extra ocr freightlens document evaluate --config configs/base.yaml
	uv run --extra layout freightlens document train --config configs/base.yaml
vision:
	uv run --extra vision freightlens vision train --config configs/base.yaml
	uv run python scripts/generate_tracking_videos.py --config configs/base.yaml
	uv run --extra vision python scripts/run_tracking.py --config configs/base.yaml
reconcile:
	uv run python scripts/generate_scenarios.py --config configs/base.yaml
	uv run python scripts/run_reconciliation.py --config configs/base.yaml
demo:
	uv run freightlens demo --config configs/demo.yaml
report:
	uv run freightlens report --config configs/smoke.yaml
api:
	uv run --no-sync uvicorn freightlens.api.app:app --host 127.0.0.1 --port 8000

model-smoke:
	uv run --extra ocr --extra vision --extra layout python scripts/model_smoke.py
