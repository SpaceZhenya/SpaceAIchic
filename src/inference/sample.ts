/**
 * Autoregressive text generation for the GPT model.
 */

import { GPT } from "../model/gpt.js";
import { makeRng } from "../training/data.js";

export interface SampleOptions {
  temperature?: number;
  topK?: number;
  seed?: number;
}

export interface SampleResult {
  /** Token ids generated (not including the prompt). */
  ids: number[];
  /** Decoded text (UTF-8). */
  text: string;
}

/** Sample one token from a raw logits row considering temperature + top-k. */
export function sampleToken(
  logits: Float64Array,
  temperature: number,
  topK: number,
  rng: () => number,
): number {
  const n = logits.length;
  let scores: Float64Array = new Float64Array(n);
  if (temperature > 0 && temperature !== 1) {
    for (let i = 0; i < n; i++) scores[i] = logits[i] / temperature;
  } else {
    scores = logits;
  }

  // top-k filtering
  const indices: number[] = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => scores[b] - scores[a]);
  const k = Math.max(1, Math.floor(topK));
  const keep = indices.slice(0, Math.min(k, n));
  let maxS = -Infinity;
  for (const i of keep) if (scores[i] > maxS) maxS = scores[i];
  let sumExp = 0;
  for (const i of keep) scores[i] = Math.exp(scores[i] - maxS);
  for (const i of keep) sumExp += scores[i];

  let r = rng() * sumExp;
  for (const i of keep) {
    r -= scores[i];
    if (r <= 0) return i;
  }
  return keep[keep.length - 1];
}

/**
 * Generate `maxTokens` bytes from `promptBytes`.
 * At each step the last blockSize context tokens are used (windows match the
 * training distribution). Returns both raw ids and decoded text.
 */
export function generate(
  model: GPT,
  promptBytes: Iterable<number>,
  opts: SampleOptions = {},
): SampleResult {
  const temperature = opts.temperature ?? 1.0;
  const topK = opts.topK ?? 50;
  const maxTokens = 256;
  const rng = makeRng(opts.seed ?? 7);
  const blockSize = model.cfg.blockSize;

  const prompt = Array.from(promptBytes).slice(-blockSize);
  const context: number[] = [...prompt];

  // dummy targets (forward needs labels but only logits are used here)
  const targets = new Int32Array(blockSize);

  for (let step = 0; step < maxTokens; step++) {
    const input = new Int32Array(blockSize);
    const start = Math.max(0, context.length - blockSize);
    for (let i = 0; i < blockSize; i++) {
      const si = start + i;
      input[i] = si < context.length ? context[si] : 0;
    }
    const { caches } = model.forward(input, targets);
    const lastCache = caches[0];
    const logits = lastCache.logits.slice(
      (blockSize - 1) * model.v,
      blockSize * model.v,
    );
    const id = sampleToken(logits, temperature, topK, rng);
    context.push(id);
  }

  const ids = context.slice(prompt.length);
  return { ids, text: new TextDecoder().decode(new Uint8Array(ids)) };
}