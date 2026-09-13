/**
 * Browser UI for SpaceAI: a BPE tokenizer trainer + an in-browser GPT trainer.
 *
 * This module is bundled with esbuild and inlined into template.html by
 * scripts/build-page.mjs, producing a single self-contained index.html that
 * runs entirely offline (CPU training in the browser).
 */

import { BPETokenizer } from "../tokenizer/index.js";
import { GPT } from "../model/gpt.js";
import { trainModel } from "../training/trainer.js";
import { generate } from "../inference/sample.js";
import { DEFAULT_CORPUS } from "../training/default-corpus.js";

function $id(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error("no element #" + id);
  return el;
}
function $v(id: string): HTMLInputElement {
  return $id(id) as HTMLInputElement;
}
const setText = (id: string, msg: string): void => {
  $id(id).textContent = msg;
};
const setProgress = (id: string, pct: number): void => {
  ($id(id) as HTMLDivElement).style.width = pct + "%";
};

// ============ BPE tokenizer ============
const tokenizer = BPETokenizer.createWithSpecialTokens(["<|endoftext|>"]);
$v("tok-corpus").value = DEFAULT_CORPUS;

function renderTokStats(): void {
  setText(
    "tok-stats",
    [
      `<span class="stat"><strong>Vocab:</strong> ${tokenizer.vocabularySize}</span>`,
      `<span class="stat"><strong>Merges:</strong> ${Math.max(0, tokenizer.vocabularySize - 256 - tokenizer.specialTokenCount)}</span>`,
      `<span class="stat"><strong>Special:</strong> ${tokenizer.specialTokenCount}</span>`,
    ].join(""),
  );
}

$id("tok-train").addEventListener("click", async () => {
  const corpus = $v("tok-corpus").value;
  const vocabSize = parseInt($v("tok-vocab").value, 10);
  if (!corpus.trim()) { setText("tok-status", "Введите корпус"); return; }
  setText("tok-status", "Обучение…");
  ($id("tok-train") as HTMLButtonElement).disabled = true;
  setProgress("tok-progress", 0);
  const start = performance.now();
  try {
    await tokenizer.train(corpus, {
      vocabularySize: vocabSize,
      onProgress: (p) => {
        setProgress("tok-progress", Math.min(100, (p.mergesMade / Math.max(1, vocabSize - 256)) * 100));
        setText("tok-status", `Слияний ${p.mergesMade}…`);
      },
      yieldEvery: 40,
    });
    setText("tok-status", `Готово за ${((performance.now() - start) / 1000).toFixed(1)}с`);
    setProgress("tok-progress", 100);
    renderTokStats();
  } catch (e) {
    setText("tok-status", "Ошибка: " + (e as Error).message);
  }
  ($id("tok-train") as HTMLButtonElement).disabled = false;
});

$id("tok-encode").addEventListener("click", () => {
  const ids = tokenizer.encode($v("tok-text").value);
  setText("tok-output", JSON.stringify(Array.from(ids)));
  renderTokStats();
});

$id("tok-decode").addEventListener("click", () => {
  const raw = $v("tok-text").value.trim();
  let ids: number[];
  try {
    ids = JSON.parse(raw);
  } catch {
    setText("tok-output", "Ошибка: нужен JSON-массив чисел (например [72,105,33])");
    return;
  }
  setText("tok-output", tokenizer.decode(ids));
  renderTokStats();
});

// ============ GPT model ============
$v("m-corpus").value = DEFAULT_CORPUS;

interface Preset {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmb: number;
}

const PRESETS: Record<string, Preset> = {
  fast: { vocabSize: 256, blockSize: 32, nLayer: 1, nHead: 2, nEmb: 32 },
  tiny: { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmb: 64 },
  small: { vocabSize: 256, blockSize: 64, nLayer: 3, nHead: 4, nEmb: 128 },
};

let model: GPT | null = null;
let trainingAbort: AbortController | null = null;
let trained = false;

function readPreset(): Preset {
  return { ...PRESETS[$v("m-preset").value] };
}

function makeModel(): GPT {
  return new GPT(readPreset());
}

function renderModelStats(): void {
  if (!model) { setText("m-stats", ""); return; }
  setText(
    "m-stats",
    [
      `<span class="stat"><strong>Params:</strong> ${model.paramCount.toLocaleString()}</span>`,
      `<span class="stat"><strong>Layers:</strong> ${model.cfg.nLayer}</span>`,
      `<span class="stat"><strong>Emb:</strong> ${model.cfg.nEmb}</span>`,
      `<span class="stat"><strong>Heads:</strong> ${model.cfg.nHead}</span>`,
      `<span class="stat"><strong>Block:</strong> ${model.cfg.blockSize}</span>`,
    ].join(""),
  );
}

function sampleLine<T>(arr: T[], count: number): T[] {
  if (arr.length <= count) return [...arr];
  const out: T[] = [];
  for (let i = 0; i < count; i++) out.push(arr[Math.floor((i / count) * arr.length)]);
  return out;
}

function drawChart(losses: number[]): void {
  const canvas = $id("m-chart") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (losses.length < 2) {
    ctx.fillStyle = "#6b7c8a";
    ctx.font = "13px ui-monospace, monospace";
    ctx.fillText("loss ещё нет — запустите обучение", 12, H / 2 + 5);
    return;
  }
  const pts = sampleLine(losses, 200);
  const minV = Math.min(...pts, 0), maxV = Math.max(...pts);
  const span = Math.max(1e-9, maxV - minV) * 1.15;
  const lo = maxV - span;
  ctx.strokeStyle = "#00d4aa";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((v, i) => {
    const x = 6 + (i / (pts.length - 1)) * (W - 12);
    const y = H - 8 - ((v - lo) / span) * (H - 20);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#6b7c8a";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`loss ${pts[0].toFixed(2)} → ${pts[pts.length - 1].toFixed(2)}`, 12, 14);
}

$id("m-train").addEventListener("click", async () => {
  const trainBtn = $id("m-train") as HTMLButtonElement;
  const stopBtn = $id("m-stop") as HTMLButtonElement;
  model = makeModel();
  trained = false;
  renderModelStats();
  trainBtn.disabled = true;
  stopBtn.disabled = false;
  ($id("m-generate") as HTMLButtonElement).disabled = true;
  const losses: number[] = [];
  trainingAbort = new AbortController();
  const corpus = $v("m-corpus").value;
  const steps = parseInt($v("m-steps").value, 10);
  const batchSize = parseInt($v("m-batch").value, 10);
  setProgress("m-progress", 0);
  setText("m-status", `Обучение: ${model.paramCount.toLocaleString()} параметров, ${steps} шагов…`);
  const start = performance.now();
  try {
    await trainModel(model, corpus, {
      corpus,
      batchSize,
      steps,
      yieldEvery: 1,
      signal: trainingAbort.signal,
      training: { learningRate: 1e-3 },
    }, (r) => {
      losses.push(r.loss);
      drawChart(losses);
      setProgress("m-progress", (r.step / r.total) * 100);
      setText("m-status", `Шаг ${r.step}/${r.total}  loss=${r.loss.toFixed(3)}  (${((performance.now() - start) / 1000).toFixed(1)}с)`);
    });
    trained = true;
    renderModelStats();
    ($id("m-generate") as HTMLButtonElement).disabled = false;
    if (trainingAbort.signal.aborted) {
      setText("m-status", `Остановлено после ${losses.length} шагов. loss=${losses[losses.length - 1].toFixed(3)}`);
    } else {
      setText("m-status", `Готово за ${((performance.now() - start) / 1000).toFixed(1)}с. loss=${losses[losses.length - 1]?.toFixed(3)}`);
      setProgress("m-progress", 100);
    }
  } catch (e) {
    setText("m-status", "Ошибка: " + (e as Error).message);
  } finally {
    trainBtn.disabled = false;
    stopBtn.disabled = true;
    trainingAbort = null;
  }
});

$id("m-stop").addEventListener("click", () => {
  const current = trainingAbort ?? new AbortController();
  current.abort();
  setText("m-status", "Остановка… (после текущего шага)");
});

$id("m-generate").addEventListener("click", () => {
  if (!model || !trained) return;
  const prompt = $v("m-prompt").value;
  const temperature = parseFloat($v("m-temp").value) || 0.7;
  const topK = parseInt($v("m-topk").value, 10) || 40;
  const stream = ($id("m-stream") as HTMLInputElement).checked;
  const output = $id("m-output") as HTMLDivElement;
  const seed = Math.floor(Math.random() * 1e9);

  if (!stream) {
    const res = generate(model, new TextEncoder().encode(prompt), { temperature, topK, seed });
    output.textContent = res.text;
    return;
  }

  output.textContent = "";
  const out = generate(model, new TextEncoder().encode(prompt), { temperature, topK, seed });
  const chunkSize = 128;
  let i = 0;
  const print = (): void => {
    const next = Math.min(i + chunkSize, out.ids.length);
    const bytes = out.ids.slice(i, next);
    output.textContent += new TextDecoder().decode(new Uint8Array(bytes));
    i = next;
    if (i < out.ids.length) setTimeout(print, 20);
  };
  setTimeout(print, 20);
});

renderTokStats();
renderModelStats();