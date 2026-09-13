/** Configuration for the GPT-style decoder-only transformer. */

export interface GPTConfig {
  /** Vocabulary size (number of distinct tokens). */
  vocabSize: number;
  /** Maximum context length / block size. */
  blockSize: number;
  /** Number of residual transformer blocks. */
  nLayer: number;
  /** Number of attention heads (must divide nEmb evenly). */
  nHead: number;
  /** Embedding / hidden dimension. */
  nEmb: number;
  /** MLP hidden multiplier. */
  mlpScale?: number;
  /** Dropout (residual connection dropout); 0 disables it. */
  dropout?: number;
  /** RMSNorm epsilon. */
  normEps?: number;
}

export function defaultGPTConfig(partial: Partial<GPTConfig>): GPTConfig {
  const base: Required<Omit<GPTConfig, "vocabSize" | "blockSize" | "nLayer" | "nHead" | "nEmb">> = {
    mlpScale: 4,
    dropout: 0.0,
    normEps: 1e-5,
  };
  const cfg: GPTConfig = {
    vocabSize: partial.vocabSize ?? 256,
    blockSize: partial.blockSize ?? 64,
    nLayer: partial.nLayer ?? 2,
    nHead: partial.nHead ?? 4,
    nEmb: partial.nEmb ?? 64,
    ...base,
    ...partial,
  };
  return cfg;
}

/** Tuneable hyperparemeters used to train the model. */
export interface TrainingConfig {
  /** Learning rate. */
  learningRate: number;
  /** AdamW weight decay. */
  weightDecay: number;
  /** AdamW beta1. */
  beta1: number;
  /** AdamW beta2. */
  beta2: number;
  /** AdamW epsilon. */
  eps: number;
  /** Gradient clip norm (0 disables). */
  gradClip: number;
}

export function defaultTrainingConfig(
  partial: Partial<TrainingConfig> = {},
): TrainingConfig {
  return { learningRate: 3e-3, weightDecay: 1e-3, beta1: 0.9, beta2: 0.99, eps: 1e-8, gradClip: 1.0, ...partial };
}