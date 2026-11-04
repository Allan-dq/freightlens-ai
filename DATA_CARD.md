# FreightLens AI data card

All project examples use public or locally generated data. Real-data results come from LOCO
(real warehouse photos) and FUNSD (real scanned forms). No proprietary operational data is
required. Locally generated companies, forms, identifiers and shipments are fictional.

| Name | Source | Type | License / terms | Size used | Purpose | Download method | Limits |
|---|---|---|---|---|---|---|---|
| Freight documents | `document_ai/generator.py` | Locally generated | AGPL-3.0-or-later generator; original synthetic output | Smoke 30/10/10; default full 1200/200/300 | Field extraction and token labeling | Seeded generation | Four types, three layouts each; invented freight records |
| Palleset | [eusandre95/Palleset](https://huggingface.co/datasets/eusandre95/Palleset) | Public synthetic, NVIDIA Isaac Sim | CC BY 4.0, attribution to eusandre95 | Initial verified subset: 2 train, 1 val, 1 test; full configuration 200/60/60 | Pallet detection and crop compositing | `HfApi` listing, selected RGB/2D NPY pairs at a fixed revision | Warehouse simulation; no real camera validation |
| LOCO | [tum-fml/loco](https://github.com/tum-fml/loco), Mayershofer et al., ICMLA 2020 | Public real photos from five logistics facilities | CC0-1.0 | 5,097 annotated images; split by recording session: development model: train 2-3 (1,943), val 5 (877); final model: train 2, 3 and 5 (2,820); test 1 and 4 (2,277) for both | Real-world detector training and evaluation (pallet, small load carrier, stillage, forklift, pallet truck) | Official archive (769 MB, SHA-256 recorded in `data/manifests/loco.json`) plus per-subset COCO annotations | Frames within a session are near-duplicates, hence session-level splits; one camera height and five sites |
| FUNSD | [Dataset project](https://guillaumejaume.github.io/FUNSD/), [repository](https://github.com/crcresearch/FUNSD) | Public real scanned forms | Upstream research dataset; no blanket commercial license is inferred from public accessibility | All 199 forms: official 149 train (134 train + 15 validation) / 50 test | LayoutLMv3 fine-tuning benchmark and qualitative OCR validation | Bounded ZIP download from original dataset site | Not freight-specific; annotation and OCR reading order differ |
| Tracking videos | `vision/video_generator.py` | Locally generated controlled sequences | Drawn fixtures generated locally; public pallet crops retain CC BY 4.0 attribution | Smoke two 4-second videos; demo seven scenarios; full 20 videos | Line/zone events and identity/count tests | Deterministic trajectories with frame-level GT | Rectangular composites and idealized motion; synthetic backgrounds |
| Reconciliation cases | `reconciliation/scenarios.py` | Locally generated state interventions | Original generated examples | Smoke 30; demo 100; full 2000 | Rules, abstention and upstream error analysis | Seeded interventions using document records and recorded visual counts | Not naturally paired warehouse shipments; aligned count interventions are explicit |

## Provenance and splits

`data/manifests/palleset_subset.json` records the source revision, seed, IDs, filenames, SHA-256
digests and exact byte count. The verified four-image subset uses 11,778,390 bytes. Upstream
training images stay in training; upstream validation is subdivided into our validation/test
sets. The downloader refuses a different requested split in the same data root. It lists all
filenames but fetches only RGB and 2D annotations; no depth, segmentation or 3D files are fetched.

The full default subset has a preflight 1,800 MiB download cap. If external access fails, rerun
`make smoke`: locally generated fixtures keep all downstream engineering paths available. There
is no automatic fallback to a whole-dataset download, and fixture data are never labeled Palleset.

Document JSON contains split, seed, template, fields, word labels, clean/noisy IDs, degradation
parameters, homography and transformed word boxes. Clean images and multi-page PDFs are retained.
Changing the configured document split or seed requires a new data root. Train/val/test sets are
disjoint by ID; layouts are shared across splits, so this is not an unseen-template benchmark.

## Data transformations

Palleset structured NPY arrays are loaded with `allow_pickle=False`. Visible bounding boxes are
clipped to the image boundary and empty boxes removed before YOLO normalization. Each conversion
validates geometry and image readability. The manifest and source annotations remain unchanged.
Only training images provide crops for generated videos. FUNSD ZIP extraction rejects path
traversal, excludes macOS metadata and caps compressed and uncompressed size.

## Attribution

Palleset: **eusandre95**, *Palleset - A Synthetic Generated Dataset for Pallet Detection*,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Crops are resized and composited by
this repository; these changes are not endorsed by the dataset author.

FUNSD: Guillaume Jaume, Hazim Kemal Ekenel and Jean-Philippe Thiran, *FUNSD: A Dataset for Form
Understanding in Noisy Scanned Documents*, ICDAR-OST 2019. Consult the upstream terms before
redistributing or using the original scanned forms commercially. The repository does not vendor
the dataset.
