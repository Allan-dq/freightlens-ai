// FreightLens AI browser demo. Runs the LOCO-trained YOLO11n detector client-side with
// onnxruntime-web. Pre- and post-processing mirror Ultralytics ONNX inference: square
// letterbox with gray (114) padding, OpenCV-style bilinear resize, per-class NMS at IoU 0.7,
// at most 300 boxes. Reconciliation uses reconcile.js, a port of the Python rules.

/* global ort, reconcile */
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const COLORS = {
  pallet: "#22c55e",
  small_load_carrier: "#eab308",
  stillage: "#3b82f6",
  forklift: "#f97316",
  pallet_truck: "#06b6d4",
};
const LABELS = {
  pallet: "Pallet",
  small_load_carrier: "Small load carrier",
  stillage: "Stillage",
  forklift: "Forklift",
  pallet_truck: "Pallet truck",
};
const FIELD_LABELS = {
  shipment_id: "Shipment ID",
  carrier: "Carrier",
  vehicle_id: "Vehicle ID",
  pallet_count: "Pallets",
  total_weight_lb: "Weight (lb)",
};
const REASONS = {
  LOW_CONFIDENCE_VISION: "the detector's mean confidence is below the review threshold",
  LOW_CONFIDENCE_DOCUMENT: "the document extraction confidence is below its threshold",
  PARTIAL_OBSERVATION: "the camera did not give a complete count",
};
const IOU_THRESHOLD = 0.7;
const MAX_DETECTIONS = 300;

const state = {
  session: null,
  meta: null,
  metrics: {},
  live: { image: null, raw: null },
  verify: { shipment: null, image: null, raw: null },
  shipments: [],
};
const $ = (id) => document.getElementById(id);

function sourcePixels(image) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, canvas.width, canvas.height).data;
}

// Bilinear sampling positions as in OpenCV INTER_LINEAR (half-pixel centres, edge clamp),
// so the tensor matches what the Python pipeline feeds the model.
function axisWeights(sourceSize, targetSize) {
  const scale = sourceSize / targetSize;
  const index = new Int32Array(targetSize);
  const weight = new Float32Array(targetSize);
  for (let d = 0; d < targetSize; d += 1) {
    let f = (d + 0.5) * scale - 0.5;
    let s = Math.floor(f);
    f -= s;
    if (s < 0) {
      s = 0;
      f = 0;
    }
    if (s >= sourceSize - 1) {
      s = sourceSize - 2;
      f = 1;
    }
    index[d] = s;
    weight[d] = f;
  }
  return { index, weight };
}

function letterbox(image, size) {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const ratio = Math.min(size / height, size / width);
  const newWidth = Math.round(width * ratio);
  const newHeight = Math.round(height * ratio);
  const left = Math.round((size - newWidth) / 2 - 0.1);
  const top = Math.round((size - newHeight) / 2 - 0.1);
  const pixels = sourcePixels(image);
  const xs = axisWeights(width, newWidth);
  const ys = axisWeights(height, newHeight);
  const plane = size * size;
  const tensor = new Float32Array(3 * plane).fill(114 / 255);
  for (let y = 0; y < newHeight; y += 1) {
    const y0 = ys.index[y];
    const wy = ys.weight[y];
    const row0 = y0 * width;
    const row1 = (y0 + 1) * width;
    for (let x = 0; x < newWidth; x += 1) {
      const x0 = xs.index[x];
      const wx = xs.weight[x];
      const a = (row0 + x0) * 4;
      const b = (row0 + x0 + 1) * 4;
      const c = (row1 + x0) * 4;
      const d = (row1 + x0 + 1) * 4;
      const target = (y + top) * size + (x + left);
      for (let ch = 0; ch < 3; ch += 1) {
        const value =
          (pixels[a + ch] * (1 - wx) + pixels[b + ch] * wx) * (1 - wy) +
          (pixels[c + ch] * (1 - wx) + pixels[d + ch] * wx) * wy;
        tensor[ch * plane + target] = Math.round(value) / 255;
      }
    }
  }
  return { tensor, ratio, left, top };
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return inter / (union + 1e-9);
}

function decode(raw, confidence) {
  const { data, dims, geometry } = raw;
  const anchors = dims[2];
  const classes = dims[1] - 4;
  const byClass = new Map();
  for (let i = 0; i < anchors; i += 1) {
    let best = 0;
    let bestClass = 0;
    for (let c = 0; c < classes; c += 1) {
      const score = data[(4 + c) * anchors + i];
      if (score > best) {
        best = score;
        bestClass = c;
      }
    }
    if (best <= confidence) continue;
    const cx = data[i];
    const cy = data[anchors + i];
    const w = data[2 * anchors + i];
    const h = data[3 * anchors + i];
    const box = [
      (cx - w / 2 - geometry.left) / geometry.ratio,
      (cy - h / 2 - geometry.top) / geometry.ratio,
      (cx + w / 2 - geometry.left) / geometry.ratio,
      (cy + h / 2 - geometry.top) / geometry.ratio,
    ];
    if (!byClass.has(bestClass)) byClass.set(bestClass, []);
    byClass.get(bestClass).push({ box, score: best, name: state.meta.names[bestClass] });
  }
  const kept = [];
  for (const candidates of byClass.values()) {
    candidates.sort((a, b) => b.score - a.score);
    const chosen = [];
    for (const candidate of candidates) {
      if (chosen.every((other) => iou(candidate.box, other.box) <= IOU_THRESHOLD)) chosen.push(candidate);
    }
    kept.push(...chosen);
  }
  kept.sort((a, b) => b.score - a.score);
  return kept.slice(0, MAX_DETECTIONS);
}

async function infer(image) {
  const size = state.meta.input_size;
  const started = performance.now();
  const { tensor, ratio, left, top } = letterbox(image, size);
  const input = new ort.Tensor("float32", tensor, [1, 3, size, size]);
  const inferenceStart = performance.now();
  const results = await state.session.run({ [state.session.inputNames[0]]: input });
  const inferenceMs = performance.now() - inferenceStart;
  const output = results[state.session.outputNames[0]];
  return {
    raw: { data: output.data, dims: output.dims, geometry: { ratio, left, top } },
    inferenceMs,
    totalMs: performance.now() - started,
  };
}

function draw(canvas, image, detections) {
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  context.lineWidth = Math.max(2, image.naturalWidth / 320);
  for (const detection of detections) {
    context.strokeStyle = COLORS[detection.name] || "#ffffff";
    const [x0, y0, x1, y1] = detection.box;
    context.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function timing(run) {
  return `Model ${run.inferenceMs.toFixed(0)} ms · end to end ${run.totalMs.toFixed(0)} ms · in your browser (WebAssembly)`;
}

/* ---------- Live detection tab ---------- */

function redrawLive() {
  if (!state.live.raw) return;
  const confidence = Number($("confidence").value);
  $("confidence-value").textContent = confidence.toFixed(2);
  const detections = decode(state.live.raw, confidence);
  draw($("result"), state.live.image, detections);
  const counts = {};
  for (const d of detections) counts[d.name] = (counts[d.name] || 0) + 1;
  const rows = Object.keys(COLORS)
    .filter((name) => counts[name])
    .map(
      (name) =>
        `<tr><td><span class="swatch" style="background:${COLORS[name]}"></span>${LABELS[name]}</td>` +
        `<td class="num">${counts[name]}</td></tr>`,
    )
    .join("");
  $("counts").innerHTML = rows || '<tr><td colspan="2" class="muted">No objects above the threshold</td></tr>';
  $("total").textContent = detections.length;
}

async function runLive(image) {
  $("status").textContent = "Running…";
  const run = await infer(image);
  state.live = { image, raw: run.raw };
  redrawLive();
  $("status").textContent = timing(run);
}

function setupLive() {
  $("confidence").addEventListener("input", redrawLive);
  document.querySelectorAll(".thumb").forEach((button) =>
    button.addEventListener("click", async () => {
      document.querySelectorAll(".thumb").forEach((b) => b.classList.toggle("active", b === button));
      await runLive(await loadImage(button.dataset.src));
    }),
  );
  $("upload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    document.querySelectorAll(".thumb").forEach((b) => b.classList.remove("active"));
    const url = URL.createObjectURL(file);
    await runLive(await loadImage(url));
    URL.revokeObjectURL(url);
  });
}

/* ---------- Verify a shipment tab ---------- */

function formatValue(value) {
  if (value === null || value === undefined) return "—";
  return typeof value === "number" ? value.toLocaleString("en-US") : value;
}

function showShipment(shipment) {
  state.verify.shipment = shipment;
  document.querySelectorAll(".scenario").forEach((b) => b.classList.toggle("active", b.dataset.id === shipment.shipment_id));
  $("doc-image").src = shipment.image;
  $("doc-image").alt = `Synthetic ${shipment.document_type.replace(/_/g, " ")} for ${shipment.shipment_id}`;
  $("doc-type").textContent = shipment.document_type.replace(/_/g, " ");
  $("doc-fields").innerHTML = shipment.fields
    .map((f) => {
      const ok = String(f.printed) === String(f.extracted);
      return (
        `<tr><td>${FIELD_LABELS[f.name]}</td><td><b>${escapeHtml(formatValue(f.extracted))}</b></td>` +
        `<td class="num muted">${f.confidence == null ? "—" : f.confidence.toFixed(2)}</td>` +
        `<td class="num ${ok ? "ok" : "bad"}" title="${ok ? "matches the printed value" : "differs from the printed value"}">${ok ? "✓" : "✗"}</td></tr>`
      );
    })
    .join("");
}

function updateVerdict() {
  const { shipment, raw, image } = state.verify;
  if (!shipment || !raw) return;
  const threshold = state.metrics.count_threshold ?? 0.25;
  const pallets = decode(raw, threshold).filter((d) => d.name === "pallet");
  draw($("verify-canvas"), image, pallets);
  const mean = pallets.length ? pallets.reduce((s, d) => s + d.score, 0) / pallets.length : null;
  const reviewThreshold = Number($("review-threshold").value);
  $("review-threshold-value").textContent = reviewThreshold.toFixed(2);
  const observed = {
    shipment_id: shipment.shipment_id,
    observed_pallet_count: pallets.length,
    observed_vehicle_id: null,
    mean_detection_confidence: mean,
    tracking_fragmentation: null,
  };
  const result = reconcile(shipment.expected, observed, {
    document_threshold: state.metrics.document_threshold ?? 0.7,
    vision_threshold: reviewThreshold,
    fragmentation_threshold: 0.4,
  });
  $("observed-count").textContent = pallets.length;
  $("observed-conf").textContent = mean === null ? "—" : mean.toFixed(2);
  const badge = $("verdict-status");
  badge.textContent = result.status.replace(/_/g, " ");
  badge.className = `verdict ${result.status.toLowerCase()}`;
  const e = result.expected_pallet_count;
  const o = result.observed_pallet_count;
  const delta = result.pallet_delta;
  const lines = [
    `The document lists <b>${e}</b> pallets (extraction confidence ${shipment.expected.document_confidence.toFixed(2)}).`,
    `The camera counted <b>${o}</b> pallets at threshold ${threshold.toFixed(2)} (mean confidence ${mean === null ? "—" : mean.toFixed(2)}).`,
  ];
  if (result.evidence.discrepancies.length) {
    lines.push(`Count check: ${delta > 0 ? "+" : ""}${delta} pallets → <b>${result.evidence.discrepancies.join(", ")}</b>.`);
  } else {
    lines.push("Count check: the counts agree.");
  }
  if (result.evidence.review_reasons.length) {
    lines.push(
      `Routed to a person because ${result.evidence.review_reasons.map((r) => REASONS[r] || r).join(" and ")}.`,
    );
  }
  lines.push(`Anomaly score ${result.anomaly_score.toFixed(2)} (weighted evidence, not a calibrated probability).`);
  if (state.verify.custom) {
    lines.push("Ground truth is unknown for your own photo.");
  } else {
    const truth = shipment.photo_annotated_pallets;
    const paperworkRight = shipment.paperwork_error === 0;
    const flagged = result.evidence.discrepancies.length > 0;
    let check;
    if (paperworkRight && !flagged) check = "the paperwork is right and the system agrees.";
    else if (!paperworkRight && flagged) check = `the paperwork overstates the load by ${shipment.paperwork_error}, so the flag is real.`;
    else if (paperworkRight && flagged) check = `the paperwork is right, so this flag is a false alarm caused by missed or extra detections.`;
    else check = "the paperwork is wrong but the camera count happens to agree with it.";
    lines.push(`<b>Ground truth:</b> the photo's LOCO annotation has ${truth} pallets; ${check}`);
  }
  $("verdict-lines").innerHTML = lines.map((l) => `<li>${l}</li>`).join("");
  $("annotated-count").textContent = state.verify.custom ? "–" : shipment.photo_annotated_pallets;
}

async function runVerify(image, custom = false) {
  $("verify-status").textContent = "Counting pallets…";
  const run = await infer(image);
  state.verify.image = image;
  state.verify.raw = run.raw;
  state.verify.custom = custom;
  updateVerdict();
  $("verify-status").textContent = timing(run);
}

async function setupVerify() {
  state.shipments = await (await fetch("shipments.json")).json();
  $("scenarios").innerHTML = state.shipments
    .map(
      (s, i) =>
        `<button class="scenario" data-id="${s.shipment_id}">Shipment ${i + 1}<span>${s.shipment_id} · ${s.expected.expected_pallet_count} pallets</span></button>`,
    )
    .join("");
  document.querySelectorAll(".scenario").forEach((button) =>
    button.addEventListener("click", async () => {
      const shipment = state.shipments.find((s) => s.shipment_id === button.dataset.id);
      showShipment(shipment);
      await runVerify(await loadImage(shipment.photo));
    }),
  );
  $("review-threshold").addEventListener("input", updateVerdict);
  $("verify-upload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file || !state.verify.shipment) return;
    const url = URL.createObjectURL(file);
    await runVerify(await loadImage(url), true);
    URL.revokeObjectURL(url);
  });
}

/* ---------- Form understanding tab ---------- */

function showForm(record) {
  document.querySelectorAll(".form-pick").forEach((b) => b.classList.toggle("active", b.dataset.id === record.id));
  $("form-image").src = record.image;
  $("form-image").alt = `LayoutLMv3 predictions on FUNSD test form ${record.id}`;
  $("form-accuracy").textContent =
    `${(record.word_label_accuracy * 100).toFixed(0)}% of ${record.words} words labeled the same as the ground truth`;
  $("entities").innerHTML = record.entities
    .map(([kind, text]) => `<tr><td><span class="tag ${kind}">${kind}</span></td><td>${escapeHtml(text)}</td></tr>`)
    .join("");
}

async function setupForms() {
  const records = await (await fetch("forms.json")).json();
  $("form-picker").innerHTML = records
    .map((r, i) => `<button class="form-pick" data-id="${r.id}">Form ${i + 1}</button>`)
    .join("");
  document.querySelectorAll(".form-pick").forEach((button) => {
    button.addEventListener("click", () => showForm(records.find((r) => r.id === button.dataset.id)));
  });
  showForm(records[0]);
}

/* ---------- Page ---------- */

function setupTabs() {
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.addEventListener("click", async () => {
      document.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", t === tab));
      document.querySelectorAll("[role=tabpanel]").forEach((panel) => {
        panel.hidden = panel.id !== tab.getAttribute("aria-controls");
      });
      if (tab.getAttribute("aria-controls") === "tab-detect" && !state.live.raw && state.session) {
        document.querySelector(".thumb").click();
      }
    });
  });
}

async function setupMetrics() {
  try {
    state.metrics = await (await fetch("metrics.json")).json();
  } catch {
    state.metrics = {};
  }
  for (const [key, value] of Object.entries(state.metrics)) {
    document.querySelectorAll(`[data-metric="${key}"]`).forEach((element) => {
      if (typeof value === "number") element.textContent = value.toFixed(element.dataset.digits ? Number(element.dataset.digits) : 3);
    });
  }
  if (state.metrics.review_threshold != null) $("review-threshold").value = state.metrics.review_threshold;
}

async function main() {
  setupTabs();
  setupForms();
  await setupMetrics();
  await setupVerify();
  setupLive();
  ort.env.wasm.wasmPaths = ORT_DIST;
  $("verify-status").textContent = "Loading the 10 MB detector…";
  const [session, meta] = await Promise.all([
    ort.InferenceSession.create("model/detector.onnx", { executionProviders: ["wasm"] }),
    fetch("model/meta.json").then((r) => r.json()),
  ]);
  state.session = session;
  state.meta = meta;
  document.querySelector(".scenario").click();
}

main().catch((error) => {
  for (const id of ["status", "verify-status"]) $(id).textContent = `Could not start the model: ${error.message}`;
  console.error(error);
});
