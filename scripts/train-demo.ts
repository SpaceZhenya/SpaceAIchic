/**
 * Node demo: trains a tiny GPT on the built-in corpus and generates text.
 * Usage: npm run demo-model
 */

import { GPT } from "../src/model/gpt.js";
import { trainModel } from "../src/training/trainer.js";
import { generate } from "../src/inference/sample.js";
import { DEFAULT_CORPUS } from "../src/training/default-corpus.js";

async function main(): Promise<void> {
  const cfg = { vocabSize: 256, blockSize: 64, nLayer: 2, nHead: 4, nEmb: 64 };
  const model = new GPT(cfg);
  console.log(`model params: ${model.paramCount} | layers=${cfg.nLayer} emb=${cfg.nEmb} head=${cfg.nHead} block=${cfg.blockSize}`);
  console.log(`corpus bytes: ${DEFAULT_CORPUS.length}`);

  const t0 = Date.now();
  const losses = await trainModel(
    model,
    DEFAULT_CORPUS,
    { corpus: DEFAULT_CORPUS, batchSize: 8, steps: 300, reportEvery: 30, yieldEvery: 0, training: { learningRate: 1e-3 } },
    (r) => {
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`step ${String(r.step).padStart(4)}/${r.total}  loss=${r.loss.toFixed(4)}  gradNorm=${r.gradNorm.toFixed(2)}  lr=${r.learningRate.toExponential(2)}  (${elapsed}s)`);
    },
  );
  console.log(`\ntrained in ${((Date.now() - t0) / 1000).toFixed(1)}s | loss ${losses[0].toFixed(3)} -> ${losses[losses.length - 1].toFixed(3)}`);

  for (const prompt of ["The fox", "The little robot", "Every morning the baker"]) {
    const out = generate(model, new TextEncoder().encode(prompt), { temperature: 0.7, topK: 40, seed: 1 });
    console.log(`\nPROMPT: ${prompt}\n---\n${out.text}\n---`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});