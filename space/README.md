---
title: FreightLens AI
emoji: 📦
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: agpl-3.0
---

# FreightLens AI demo

Two models from [FreightLens AI](https://github.com/Allan-dq/freightlens-ai), both fine-tuned on
real public data:

- **Warehouse objects**: YOLO11n fine-tuned on [LOCO](https://github.com/tum-fml/loco) (CC0),
  real images from logistics facilities. Detects pallets, small load carriers, stillages,
  forklifts and pallet trucks.
- **Form understanding**: LayoutLMv3-base fine-tuned on [FUNSD](https://guillaumejaume.github.io/FUNSD/)
  scanned forms. Tesseract OCR reads the words; the model tags questions, answers and headers.

Example images come from the LOCO and FUNSD test splits, which were not used for training.
LayoutLMv3 weights are released under CC BY-NC-SA 4.0, so this demo is for non-commercial use.
