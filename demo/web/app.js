// FreightLens AI browser demo. Runs the LOCO-trained YOLO11n detector client-side with
// onnxruntime-web. Pre- and post-processing mirror Ultralytics ONNX inference: square
// letterbox with gray (114) padding, per-class NMS at IoU 0.7, at most 300 boxes.

/* global ort */
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
const IOU_THRESHOLD = 0.7;
const MAX_DETECTIONS = 300;

const state = { session: null, meta: null, image: null, raw: null, geometry: null };
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

function decode(output, dims, geometry, confidence) {
  const channels = dims[1];
  const anchors = dims[2];
  const classes = channels - 4;
  const byClass = new Map();
  for (let i = 0; i < anchors; i += 1) {
    let best = 0;
    let bestClass = 0;
    for (let c = 0; c < classes; c += 1) {
      const score = output[(4 + c) * anchors + i];
      if (score > best) {
        best = score;
        bestClass = c;
      }
    }
    if (best <= confidence) continue;
    const cx = output[i];
    const cy = output[anchors + i];
    const w = output[2 * anchors + i];
    const h = output[3 * anchors + i];
    const box = [
      (cx - w / 2 - geometry.left) / geometry.ratio,
      (cy - h / 2 - geometry.top) / geometry.ratio,
      (cx + w / 2 - geometry.left) / geometry.ratio,
      (cy + h / 2 - geometry.top) / geometry.ratio,
    ];
    if (!byClass.has(bestClass)) byClass.set(bestClass, []);
    byClass.get(bestClass).push({ box, score: best, cls: bestClass });
  }
  const kept = [];
  for (const candidates of byClass.values()) {
    candidates.sort((a, b) => b.score - a.score);
    const chosen = [];
    for (const candidate of candidates) {
      if (chosen.every((other) => iou(candidate.box, other.box) <= IOU_THRESHOLD)) {
        chosen.push(candidate);
      }
    }
    kept.push(...chosen);
  }
  kept.sort((a, b) => b.score - a.score);
  return kept.slice(0, MAX_DETECTIONS);
}

function render(detections) {
  const canvas = $("result");
  const image = state.image;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  context.lineWidth = Math.max(2, image.naturalWidth / 320);
  const counts = {};
  for (const detection of detections) {
    const name = state.meta.names[detection.cls];
    counts[name] = (counts[name] || 0) + 1;
    context.strokeStyle = COLORS[name] || "#ffffff";
    const [x0, y0, x1, y1] = detection.box;
    context.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
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

function redraw() {
  if (!state.raw) return;
  const confidence = Number($("confidence").value);
  $("confidence-value").textContent = confidence.toFixed(2);
  render(decode(state.raw.data, state.raw.dims, state.geometry, confidence));
}

async function run(image) {
  state.image = image;
  $("status").textContent = "Running…";
  const size = state.meta.input_size;
  const started = performance.now();
  const { tensor, ratio, left, top } = letterbox(image, size);
  const input = new ort.Tensor("float32", tensor, [1, 3, size, size]);
  const inferenceStart = performance.now();
  const results = await state.session.run({ [state.session.inputNames[0]]: input });
  const inferenceMs = performance.now() - inferenceStart;
  const output = results[state.session.outputNames[0]];
  state.raw = { data: output.data, dims: output.dims };
  state.geometry = { ratio, left, top };
  redraw();
  const totalMs = performance.now() - started;
  $("status").textContent = `Model ${inferenceMs.toFixed(0)} ms · end to end ${totalMs.toFixed(0)} ms · in your browser (WebAssembly)`;
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

async function useExample(button) {
  document.querySelectorAll(".thumb").forEach((b) => b.classList.toggle("active", b === button));
  await run(await loadImage(button.dataset.src));
}

function setupTabs() {
  document.querySelectorAll("[role=tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll("[role=tab]").forEach((t) => t.setAttribute("aria-selected", t === tab));
      document.querySelectorAll("[role=tabpanel]").forEach((panel) => {
        panel.hidden = panel.id !== tab.getAttribute("aria-controls");
      });
    });
  });
}

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

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
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

async function setupMetrics() {
  try {
    const metrics = await (await fetch("metrics.json")).json();
    for (const [key, value] of Object.entries(metrics)) {
      const element = document.querySelector(`[data-metric="${key}"]`);
      if (element && typeof value === "number") element.textContent = value.toFixed(3);
    }
  } catch {
    /* metrics are optional decoration */
  }
}

async function main() {
  setupTabs();
  setupForms();
  setupMetrics();
  ort.env.wasm.wasmPaths = ORT_DIST;
  $("status").textContent = "Loading the 10 MB model…";
  const [session, meta] = await Promise.all([
    ort.InferenceSession.create("model/detector.onnx", { executionProviders: ["wasm"] }),
    fetch("model/meta.json").then((r) => r.json()),
  ]);
  state.session = session;
  state.meta = meta;
  $("confidence").addEventListener("input", redraw);
  document.querySelectorAll(".thumb").forEach((button) => button.addEventListener("click", () => useExample(button)));
  $("upload").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    document.querySelectorAll(".thumb").forEach((b) => b.classList.remove("active"));
    const url = URL.createObjectURL(file);
    await run(await loadImage(url));
    URL.revokeObjectURL(url);
  });
  await useExample(document.querySelector(".thumb"));
}

main().catch((error) => {
  $("status").textContent = `Could not start the model: ${error.message}`;
  console.error(error);
});
