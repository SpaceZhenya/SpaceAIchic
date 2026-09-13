/**
 * Byte-level BPE (Byte Pair Encoding) trainer.
 *
 * Strategy (matches the "basic" GPT-2-style byte-level BPE):
 *  - the corpus is normalized, pretokenized, and flattened into one stream of
 *    UTF-8 byte ids (0-255);
 *  - adjacent byte pairs are counted and repeatedly merged, most frequent
 *    first, until the requested vocabulary size is reached (or no useful pair
 *    remains);
 *  - merge order is preserved: the first merged pair is preferred in the
 *    encoder, exactly like the original GPT-2 tokenizer.
 *
 * Counting uses an incremental pair-frequency map so a merge only touches the
 * neighbourhood of each merge site instead of re-scanning the whole corpus.
 */

import { normalizeText, pretokenize, encodeBytes } from "./unicode.js";

/** Base vocabulary: one id per possible byte value. */
export const BASE_VOCABULARY = 256;

/** Multiplier used to pack two ids into one numeric map key. */
export const PAIR_MULTIPLIER = 1_000_000;

/** Pack two adjacent token ids into a single numeric key. */
export function pairKey(a: number, b: number): number {
  return a * PAIR_MULTIPLIER + b;
}

/** Unpack a pair key into [a, b]. */
export function unpackPair(key: number): [number, number] {
  return [Math.floor(key / PAIR_MULTIPLIER), key % PAIR_MULTIPLIER];
}

export interface TrainResult {
  /** Merged pair -> token id (>= BASE_VOCABULARY). */
  merges: Map<number, number>;
  /** mergePairs[i] is the pair that produced token id BASE_VOCABULARY + i. */
  mergePairs: Array<[number, number]>;
  /** How many merges were actually performed. */
  mergesMade: number;
  /** Total number of byte tokens seen (corpus length). */
  corpusTokens: number;
}

export interface TrainProgress {
  mergeIndex: number;
  mergesMade: number;
  currentLength: number;
}

export interface TrainOptions {
  /** Requested total vocabulary size (256 + merges). */
  vocabularySize: number;
  /** Optional progress callback for long trainings. */
  onProgress?: (progress: TrainProgress) => void;
  /** Call the progress callback at most once every N merges. */
  progressEvery?: number;
  /**
   * Yield control to the event loop (setTimeout 0) every N merges so a browser
   * tab stays responsive during long trainings. Default: 0 (no yielding).
   */
  yieldEvery?: number;
}

function mergeWithPositions(
  ids: number[],
  a: number,
  b: number,
  newId: number,
): { ids: number[]; mergePositions: number[]; occurrenceStarts: number[] } {
  const next: number[] = [];
  const mergePositions: number[] = [];
  const occurrenceStarts: number[] = [];
  let i = 0;
  while (i < ids.length) {
    if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) {
      occurrenceStarts.push(i);
      next.push(newId);
      mergePositions.push(next.length - 1);
      i += 2;
    } else {
      next.push(ids[i]);
      i += 1;
    }
  }
  return { ids: next, mergePositions, occurrenceStarts };
}

/**
 * Recompute the pair-frequency map locally after a single merge.
 *
 * Correctness relies on position-based tracking: the only pairs that change
 * are the ones touching a merge site. Decrements use OLD-array positions,
 * increments use NEW-array positions, and the "start position" space is
 * disjoint enough that every pair is counted exactly once.
 */
function updateStats(
  stats: Map<number, number>,
  oldIds: number[],
  newIds: number[],
  occurrenceStarts: number[],
  mergePositions: number[],
  newId: number,
): void {
  // Pairs in the OLD array that are destroyed or changed.
  const affectedOld = new Set<number>();
  for (const i of occurrenceStarts) {
    if (i - 1 >= 0) affectedOld.add(i - 1);
    if (i >= 0 && i < oldIds.length - 1) affectedOld.add(i);
    if (i + 1 < oldIds.length - 1) affectedOld.add(i + 1);
  }
  for (const p of affectedOld) {
    decrement(stats, pairKey(oldIds[p], oldIds[p + 1]));
  }

  // Pairs in the NEW array that are created next to merged tokens.
  // Right neighbours are always fresh; a left neighbour that is itself a
  // freshly merged token was already counted as the previous right neighbour.
  for (const j of mergePositions) {
    if (j + 1 < newIds.length) {
      increment(stats, pairKey(newIds[j], newIds[j + 1]));
    }
    if (j - 1 >= 0 && newIds[j - 1] !== newId) {
      increment(stats, pairKey(newIds[j - 1], newIds[j]));
    }
  }
}

function increment(stats: Map<number, number>, key: number): void {
  stats.set(key, (stats.get(key) ?? 0) + 1);
}

function decrement(stats: Map<number, number>, key: number): void {
  const next = (stats.get(key) ?? 0) - 1;
  if (next <= 0) {
    stats.delete(key);
  } else {
    stats.set(key, next);
  }
}

function countPairs(ids: number[]): Map<number, number> {
  const stats = new Map<number, number>();
  for (let i = 0; i + 1 < ids.length; i++) {
    increment(stats, pairKey(ids[i], ids[i + 1]));
  }
  return stats;
}

function mostFrequentPair(stats: Map<number, number>): [number, number] | null {
  let bestKey = -1;
  let bestCount = 0;
  for (const [key, count] of stats) {
    if (count > bestCount) {
      bestCount = count;
      bestKey = key;
    }
  }
  return bestKey === -1 ? null : [bestKey, bestCount];
}

/**
 * Train byte-level BPE over a raw corpus string.
 *
 * `options.vocabularySize` is the TOTAL vocabulary size including the 256
 * base byte tokens. Training stops early when no pair occurs twice.
 */
export async function trainBPE(
  corpus: string,
  options: TrainOptions,
): Promise<TrainResult> {
  const {
    vocabularySize,
    onProgress,
    progressEvery = 1,
    yieldEvery = 0,
  } = options;
  const maxMerges = vocabularySize - BASE_VOCABULARY;
  if (maxMerges < 0) {
    throw new Error(`vocabularySize must be >= ${BASE_VOCABULARY}`);
  }

  const normalized = normalizeText(corpus);
  const pieces = pretokenize(normalized);
  const ids: number[] = [];
  for (const piece of pieces) {
    for (const byte of encodeBytes(piece)) {
      ids.push(byte);
    }
  }
  const corpusTokens = ids.length;

  const merges = new Map<number, number>();
  const mergePairs: Array<[number, number]> = [];
  let current = ids;
  const stats = countPairs(current);

  let forwardProgress = 0;
  for (let mergeIndex = 0; mergeIndex < maxMerges; mergeIndex++) {
    const best = mostFrequentPair(stats);
    if (best === null || best[1] < 2) break;

    const [bestPair, count] = best;
    const [a, b] = unpackPair(bestPair);

    if (count < 2) break;

    const newId = BASE_VOCABULARY + mergeIndex;
    const { ids: next, mergePositions, occurrenceStarts } = mergeWithPositions(
      current,
      a,
      b,
      newId,
    );
    updateStats(stats, current, next, occurrenceStarts, mergePositions, newId);

    current = next;
    merges.set(bestPair, newId);
    mergePairs.push([a, b]);

    if (onProgress) {
      forwardProgress += 1;
      if (forwardProgress >= progressEvery) {
        forwardProgress = 0;
        onProgress({
          mergeIndex,
          mergesMade: mergePairs.length,
          currentLength: current.length,
        });
      }
    }
    if (yieldEvery > 0 && (mergeIndex % yieldEvery === 0 || mergeIndex === maxMerges - 1)) {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  return {
    merges,
    mergePairs,
    mergesMade: mergePairs.length,
    corpusTokens,
  };
}