# FreightLens AI model card

## Intended use

A reproducible research demonstration of expected-versus-observed freight
verification. It is not validated for production warehouse decisions, security surveillance,
inventory accounting or unattended exception resolution.

## Components

| Component | Source | Training / behavior | License |
|---|---|---|---|
| OCR | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Pretrained English PP-OCRv6 medium detection/recognition, CPU inference; no local OCR training | Apache-2.0 project; upstream model terms apply |
| Field baseline | This repository | Deterministic labels, spatial grouping and numeric/unit validation | AGPL-3.0-or-later |
| Form model | [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) | Full fine-tuning on FUNSD (1,000 steps, AdamW 1e-5, linear warm-up), best checkpoint by validation F1; the frozen-backbone head on generated freight forms is kept as a separate experiment | **CC BY-NC-SA 4.0** pretrained model; noncommercial restriction remains |
| Warehouse detector | [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/) | Pretrained COCO initialization, fine-tuned on LOCO real warehouse photos (5 classes) with session-level splits; the final model trains on sessions 2, 3 and 5 for the 37 epochs selected by the development run; Palleset single-class path kept for offline smoke | Ultralytics AGPL-3.0 / separate commercial licensing |
| Tracking | [Ultralytics ByteTrack](https://docs.ultralytics.com/modes/track/) | Explicit `bytetrack.yaml`, persistent state per video, reset between videos | Ultralytics integration license applies |
| Offline fixtures | This repository | Seeded OCR token corruption and pixel-color/centroid tracker; no learned-model claim | AGPL-3.0-or-later |
| Reconciliation | This repository | Count/vehicle rules, transparent weighted score, review policy | AGPL-3.0-or-later |

The project source license does not relicense external datasets or weights. LayoutLMv3's
noncommercial restriction is material to reuse. Model weights and downloaded data are not
committed to this repository.

## Evaluation and hardware

Real-data training (LOCO detector, FUNSD LayoutLMv3) ran on an Apple M1 Pro with the PyTorch MPS
backend; settings are in `configs/real.yaml` and each run is recorded under `artifacts/runs/`.

Local verification used Python 3.11 on a Windows Intel Core i7-12700K CPU, without a GPU. Exact
package versions are in `uv.lock`; every recorded experiment includes installed package versions,
config, seed, UTC timestamp and Git revision when available under `artifacts/runs/`.

Smoke model training intentionally uses only two training images and one epoch for YOLO and one
tiny optimizer step for LayoutLMv3. The resulting checkpoints establish executable integration,
not useful final accuracy. Their actual measured metrics are saved without substituting targets
or benchmark numbers. Full training uses the larger configured splits and separate output roots.

The LayoutLMv3 experiment uses labeled ground-truth words/boxes. Its entity F1 is **not directly
comparable** to rule-parser field F1 or raster OCR accuracy. A new token head may predict no
entities after the smoke step; a zero score is a valid measured result.

The COCO model has no pallet class. The detector wrapper refuses to silently interpret COCO
classes as pallets and requires fine-tuned weights for pallet inference. A small initialization
experiment can miss every pallet at the counting confidence threshold even when low-threshold
AP is nonzero.

## Confidence and limits

OCR confidence, label/spatial validity and field availability feed a heuristic document score.
Detection confidence and optional known fragmentation feed review rules. The anomaly score is
an interpretable weighted score, not a calibrated probability. Unknown counts remain `null`;
unknown plates do not become a vehicle match. Review results retain discrepancy evidence but
do not assert a confirmed anomaly.

There is no proprietary operational data, real warehouse deployment, business-impact estimate,
plate recognition or multi-camera identity model. Synthetic forms, simulated imagery and
controlled trajectories leave significant domain gaps. Results may degrade on unseen forms,
small or occluded pallets, crowded scenes and unfamiliar cameras. Fragmentation metrics use
ground truth during evaluation; inference never pretends to know ground-truth fragmentation.
