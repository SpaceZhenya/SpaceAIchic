import { describe, it, expect } from "vitest";
import { GPT, AdamW } from "../src/model/gpt.js";
import { defaultTrainingConfig } from "../src/model/config.js";
import { trainModel } from "../src/training/trainer.js";
import { generate } from "../src/inference/sample.js";

function tinyConfig() {
  return { vocabSize: 16, blockSize: 4, nLayer: 2, nHead: 2, nEmb: 8 };
}

function randomIds(n: number, vocab: number, seed = 1): Int32Array {
  let s = seed;
  const rng = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return (s % vocab + vocab) % vocab;
  };
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = rng();
  return out;
}

describe("GPT forward/backward", () => {
  it("computes finite loss within reason", () => {
    const model = new GPT(tinyConfig());
    const T = model.cfg.blockSize;
    const x = randomIds(2 * T, model.v);
    const y = randomIds(2 * T, model.v);
    const { loss } = model.forward(x, y);
    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(Math.log(model.v) + 0.5);
  });

  it("gradients match numerical gradients", () => {
    const model = new GPT(tinyConfig());
    const T = model.cfg.blockSize;
    const x = randomIds(2 * T, model.v, 7);
    const y = randomIds(2 * T, model.v, 8);
    const { caches } = model.forward(x, y);
    model.zeroGrads();
    model.backward(x, y, caches);

    // pick a sample of parameters to check
    const checks: Array<{ name: string; index: number }> = [];
    for (const name of model.order) {
      const w = model.weights.get(name)!;
      const step = Math.max(1, Math.floor(w.length / 8));
      for (let i = 0; i < w.length; i += step) checks.push({ name, index: i });
    }

    const eps = 1e-4;
    let maxRel = 0;
    for (const { name, index } of checks) {
      const w = model.weights.get(name)!;
      const orig = w[index];

      w[index] = orig - eps;
      const l0 = model.forward(x, y).loss;
      w[index] = orig + eps;
      const l1 = model.forward(x, y).loss;
      w[index] = orig;

      const numGrad = (l1 - l0) / (2 * eps);
      const anaGrad = model.grads.get(name)![index];
      const rel = Math.abs(anaGrad - numGrad) / (Math.abs(numGrad) + 1e-9);
      maxRel = Math.max(maxRel, rel);
      expect(rel).toBeLessThan(0.01);
    }
    void maxRel;
  });

  it("model stores the expected number of parameters", () => {
    const model = new GPT(tinyConfig());
    expect(model.paramCount).toBeGreaterThan(1000);
    expect(model.order.length).toBe(model.order.filter((n, i, a) => a.indexOf(n) === i).length);
  });
});

describe("training", () => {
  it("reduces loss over training steps", async () => {
    const model = new GPT({ vocabSize: 256, blockSize: 16, nLayer: 2, nHead: 2, nEmb: 16 });
    const corpus = "the fox and the crow the fox and the crow the fox and the crow little robot says hello.";
    const history: number[] = [];
    await trainModel(
      model,
      corpus,
      { corpus, batchSize: 4, steps: 120, reportEvery: 20, yieldEvery: 0, training: { learningRate: 5e-3 } },
      (r) => history.push(r.loss),
    );
    expect(history.length).toBeGreaterThanOrEqual(5);
    const first = history.slice(0, 2).reduce((a, b) => a + b, 0) / 2;
    const last = history.slice(-2).reduce((a, b) => a + b, 0) / 2;
    expect(last).toBeLessThan(first);
    expect(last).toBeLessThan(Math.log(256) * 0.9);
  });
});

describe("sampling", () => {
  it("generates valid decodable output", () => {
    const model = new GPT({ vocabSize: 256, blockSize: 8, nLayer: 1, nHead: 2, nEmb: 8 });
    const result = generate(model, [116, 104, 101], { temperature: 0.8, topK: 20 });
    expect(result.ids.length).toBeGreaterThan(0);
    expect(result.ids.every((id) => id >= 0 && id < 256)).toBe(true);
    expect(typeof result.text).toBe("string");
    expect(result.text.length).toBeGreaterThan(0);
  });
});