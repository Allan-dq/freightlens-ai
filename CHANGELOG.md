# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- **Final detector** trained on LOCO sessions 2, 3 and 5 for the 37 epochs chosen on validation
  (`configs/real_final.yaml`, last checkpoint). Test pallet AP50 0.551 (development model 0.522),
  mAP50 0.303, count MAE 6.7. Weights, model card and the browser demo now use it.
- **Tiled inference experiment** (`scripts/evaluate_tiling.py`) with an evaluator that reproduces
  the Ultralytics AP. About twice the resolution adds 0.057 pallet AP50 on validation but only
  0.012 on test.
- `scripts/evaluate_onnx.py` exports the detector and scores the export on the test split.
- `make ci-local` runs lint, mypy and tests without the optional extras, as CI does.
- The verify tab anchors each document to the photo's annotated pallet count and states whether
  a flag is real or a false alarm.

### Fixed

- Tiling evaluation fails clearly on unreadable images (found by mypy in the dev-only CI
  environment).

## [1.0.0] - 2026-11-04

First release with models trained and evaluated on real, public data.

### Added

- **Warehouse detector on LOCO.** YOLO11n fine-tuned on LOCO (CC0 real warehouse photos) with
  session-level train/validation/test splits; `freightlens data prepare-loco` and
  dataset-aware `vision train` / `vision evaluate`. Test (unseen sessions): pallet AP50 0.522,
  mAP50 over five classes 0.271.
- **Form model on FUNSD.** End-to-end LayoutLMv3 fine-tuning (`document train-funsd`) with word-
  and segment-level boxes. Test entity F1 0.797 (word boxes) and 0.878 (segment boxes).
- **Counting threshold selection.** `vision calibrate` picks the pallet-counting threshold on the
  validation session (0.25) and scores the test sessions once (MAE 7.2, bias -0.9).
- **Browser demo** on Hugging Face Spaces: shipment verification with in-browser reconciliation,
  live warehouse detection with ONNX Runtime Web, and precomputed form understanding.
  `scripts/export_web_demo.py` rebuilds its assets.
- **Reconciliation parity tests.** Shared cases check the Python rules and their JavaScript port
  (pytest and Node in CI).
- Latency benchmark (`scripts/benchmark_latency.py`), README results generated from metric files
  (`report-real`), example renderer, model publishing script with generated model cards.
- Device selection (`auto` picks CUDA, Apple MPS or CPU) and `configs/real.yaml`.

### Changed

- README leads with real-data results, error analysis and the live demo; generated-data pipeline
  checks moved to their own section.
- The Gradio app moved to `demo/gradio/` for local runs.
- API version is read from `freightlens.__version__`.

### Model weights

- [AllanDai/freightlens-yolo11n-loco](https://huggingface.co/AllanDai/freightlens-yolo11n-loco)
  (PyTorch and ONNX)
- [AllanDai/freightlens-layoutlmv3-funsd](https://huggingface.co/AllanDai/freightlens-layoutlmv3-funsd)
  (CC BY-NC-SA 4.0, non-commercial)

## [0.1.0] - 2026-09-03

Initial implementation: generated freight documents, OCR adapters and rule parser, YOLO + ByteTrack
pipeline on Palleset and controlled videos, reconciliation rules with review policy, FastAPI
service, CLI, Docker and CI.

[1.0.0]: https://github.com/Allan-dq/freightlens-ai/releases/tag/v1.0.0
[0.1.0]: https://github.com/Allan-dq/freightlens-ai/tree/5978629
