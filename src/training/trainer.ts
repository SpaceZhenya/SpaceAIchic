/**
 * Training loop for the in-browser GPT model:
 *  - builds byte-level batches from a text corpus
 *  - runs forward/backward with AdamW
 *  - linear warmup + cosine decay over the run
 *  - yields to the event loop so the browser stays responsive
 */

import { GPT, AdamW } from "../model/gpt.js";
import { defaultTrainingConfig, type TrainingConfig } from "../model/config.js";
import { ByteCorpus, makeRng } from "./data.js";

export interface TrainRunConfig {
  /** Corpus text. */
  corpus: string;
  /** Batch size (number of sequences). */
  batchSize?: number;
  /** Number of training steps. */
  steps?: number;
  /** How often to report progress (report every N steps). */
  reportEvery?: number;
  /** Random seed. */
  seed?: number;
  /** AdamW / LR hyperparameters (overrides defaults). */
  training?: Partial<TrainingConfig>;
  /** Yield to the event loop every N steps (0 disables). */
  yieldEvery?: number;
  /** Abort signal: training stops cleanly at the next step boundary. */
  signal?: AbortSignal;
}

export interface TrainReport {
  step: number;
  total: number;
  loss: number;
  gradNorm: number;
  learningRate: number;
}

/**
 * Train `model` on `corpus` text. Returns the loss history.
 * `onReport` fires every `reportEvery` steps with the average loss.
 */
export async function trainModel(
  model: GPT,
  corpus: string,
  options: TrainRunConfig,
  onReport: (r: TrainReport) => void,
): Promise<number[]> {
  const batchSize = options.batchSize ?? 4;
  const steps = options.steps ?? 200;
  const reportEvery = options.reportEvery ?? 10;
  const yieldEvery = options.yieldEvery ?? 10;
  const tcfg = defaultTrainingConfig(options.training);
  const blockSize = model.cfg.blockSize;

  const corpusData = new ByteCorpus(corpus);
  if (corpusData.bytes.length < blockSize + 2) {
    throw new Error("corpus is too short to train on");
  }
  const rng = makeRng(options.seed ?? 42);
  const optim = new AdamW(model);
  optim.reset();

  const history: number[] = [];
  // learning rate schedule: warmup to max then cosine down to ~0
  const warmupSteps = Math.min(10, Math.floor(steps * 0.1));

  for (let step = 0; step < steps; step++) {
    if (options.signal?.aborted) break;
    const { x, y } = corpusData.nextBatch(batchSize, blockSize, rng);
    model.zeroGrads();
    const { loss, caches } = model.forward(x, y);
    model.backward(x, y, caches);

    // scheduled learning rate
    const rawLr = computeLr(step, steps, warmupSteps, tcfg.learningRate);
    const norm = optim.stepOnce({ ...tcfg, learningRate: rawLr });

    history.push(loss);

    if (step % reportEvery === 0 || step === steps - 1) {
      onReport({ step: step + 1, total: steps, loss, gradNorm: norm, learningRate: rawLr });
    }
    if (yieldEvery > 0 && (step % yieldEvery === 0 || step === steps - 1)) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  return history;
}

function computeLr(step: number, total: number, warmup: number, base: number): number {
  // linear warmup
  if (step < warmup) return base * ((step + 1) / warmup);
  // cosine decay
  const progress = (step - warmup) / Math.max(1, total - warmup);
  return base * 0.5 * (1 + Math.cos(Math.PI * progress));
}