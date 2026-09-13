/**
 * Minimal byte-level data utilities for training and sampling.
 *
 * The in-browser model is intentionally byte-level (vocab 256): UTF-8 bytes
 * are the tokens. This keeps model size tiny while still being a real
 * language model over raw text.
 */

/** Encode raw text into UTF-8 byte ids (0-255). */
export function textToBytes(text: string): Int32Array {
  const enc = new TextEncoder().encode(text);
  const out = new Int32Array(enc.length);
  for (let i = 0; i < enc.length; i++) out[i] = enc[i];
  return out;
}

/** Decode byte ids back into text. */
export function bytesToText(ids: Iterable<number>): string {
  const bytes: number[] = [];
  for (const id of ids) bytes.push(id);
  return new TextDecoder().decode(new Uint8Array(bytes));
}

/** Deterministic seedable PRNG (mulberry32). */
export function makeRng(seed = 1337): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A cursor over a byte corpus that yields random contiguous blocks.
 * Uses a round-robin start position so every byte is seen eventually.
 */
export class ByteCorpus {
  readonly bytes: Int32Array;

  constructor(text: string) {
    this.bytes = textToBytes(text);
  }

  /** Sample a batch {x, y} of batchSize blocks of length blockSize (shifted). */
  nextBatch(
    batchSize: number,
    blockSize: number,
    rng: () => number,
  ): { x: Int32Array; y: Int32Array } {
    const n = this.bytes.length;
    const x = new Int32Array(batchSize * blockSize);
    const y = new Int32Array(batchSize * blockSize);
    for (let b = 0; b < batchSize; b++) {
      const start = Math.floor(rng() * n);
      for (let t = 0; t < blockSize; t++) {
        const i = (start + t) % n;
        x[b * blockSize + t] = this.bytes[i];
        y[b * blockSize + t] = this.bytes[(i + 1) % n];
      }
    }
    return { x, y };
  }
}