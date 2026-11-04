---
title: FreightLens AI
emoji: 📦
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: agpl-3.0
short_description: Warehouse object detection in your browser, trained on LOCO
---

# FreightLens AI demo

- **Warehouse detection (live):** YOLO11n fine-tuned on [LOCO](https://github.com/tum-fml/loco)
  (CC0, real logistics facilities), exported to ONNX and run client-side with ONNX Runtime Web.
  Uploaded photos never leave the browser.
- **Form understanding (precomputed):** LayoutLMv3-base fine-tuned on
  [FUNSD](https://guillaumejaume.github.io/FUNSD/) scanned forms. Outputs on test forms are
  computed ahead of time because the model and OCR are too large for a browser.

Training code, evaluation and metrics: [github.com/Allan-dq/freightlens-ai](https://github.com/Allan-dq/freightlens-ai).
LayoutLMv3 weights are CC BY-NC-SA 4.0 (non-commercial).
