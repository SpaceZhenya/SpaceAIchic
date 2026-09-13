/**
 * A byte-level BPE tokenizer with encode / decode / train / serialize.
 *
 * Vocabulary layout:
 *   0..255                    raw UTF-8 bytes
 *   256 .. 255 + merges-1     learned BPE merges (in creation order)
 *   top ids                   special tokens (e.g. <|endoftext|>)
 *
 * Encoding is greedy and rank-aware: among all mergeable adjacent pairs the
 * one trained EARLIEST wins each round, matching the original GPT-2 BPE.
 */

import {
  BASE_VOCABULARY,
  PAIR_MULTIPLIER,
  pairKey,
  trainBPE,
  type TrainProgress,
} from "./trainer.js";
import {
  normalizeText,
  pretokenize,
  encodeBytes,
  decodeBytes,
  REPLACEMENT_CHAR,
} from "./unicode.js";
import {
  DEFAULT_SPECIAL_TOKENS,
  escapeRegExp,
  validateSpecialToken,
} from "./special-tokens.js";

export interface BPEVocab {
  /** Merged pair key -> token id (>= 256). */
  merges: Array<[number, number]>;
  /** Special tokens in id order (ids are the highest in the vocabulary). */
  specialTokens: string[];
}

export interface TrainOptions {
  /** Total vocabulary size (bytes + merges + special tokens). */
  vocabularySize: number;
  /** Callback for progress reporting during long trainings. */
  onProgress?: (progress: TrainProgress) => void;
  /**
   * Yield to the event loop every N merges so a browser tab stays responsive.
   * Default: 0 (no yielding).
   */
  yieldEvery?: number;
}

export class BPETokenizer {
  private merges: Map<number, number> = new Map();
  private readonly specialTokens: readonly string[];
  private readonly specialRegex: RegExp | null;
  /** id -> raw UTF-8 byte sequence, used to decode. */
  private vocabularyBytes: Uint8Array[] = [];

  private constructor(specialTokens: readonly string[]) {
    this.specialTokens = specialTokens;
    this.vocabularyBytes = buildVocabulary(0, []);
    if (specialTokens.length === 0) {
      this.specialRegex = null;
    } else {
      const ordered = [...specialTokens].sort((x, y) => y.length - x.length);
      this.specialRegex = new RegExp(
        ordered.map((t) => escapeRegExp(t)).join("|"),
        "g",
      );
    }
  }

  /** Create an empty tokenizer without special tokens. */
  static create(): BPETokenizer {
    return new BPETokenizer([]);
  }

  /** Create a tokenizer reserving the given special tokens at the top. */
  static createWithSpecialTokens(
    specialTokens: readonly string[] = DEFAULT_SPECIAL_TOKENS,
  ): BPETokenizer {
    for (const token of specialTokens) validateSpecialToken(token);
    return new BPETokenizer(specialTokens);
  }

  /** Number of known tokens, including special tokens. */
  get vocabularySize(): number {
    return this.vocabularyBytes.length + this.specialTokens.length;
  }

  get specialTokenCount(): number {
    return this.specialTokens.length;
  }

  /**
   * Train the tokenizer on a raw corpus. `vocabularySize` is the total number
   * of tokens in the final vocabulary (bytes + merges + special tokens).
   */
  async train(corpus: string, options: TrainOptions): Promise<void> {
    const { vocabularySize, onProgress, yieldEvery } = options;
    const specialCount = this.specialTokens.length;
    const budget = vocabularySize - BASE_VOCABULARY - specialCount;
    if (budget < 0) {
      throw new Error(
        `vocabularySize is too small: need at least ${BASE_VOCABULARY + specialCount}`,
      );
    }

    const result = await trainBPE(corpus, {
      vocabularySize: BASE_VOCABULARY + budget,
      onProgress,
      yieldEvery,
    });

    this.merges = result.merges;
    this.vocabularyBytes = buildVocabulary(
      result.mergesMade,
      result.mergePairs,
    );
  }

  /** First token id that belongs to the reserved special-token block. */
  private get specialBase(): number {
    return BASE_VOCABULARY + this.merges.size;
  }

  private specialId(token: string): number {
    return this.specialBase + this.specialTokens.indexOf(token);
  }

  /** Convert normal text (no special tokens) into a token id array. */
  encodeOrdinary(text: string): number[] {
    const pieces = pretokenize(normalizeText(text));
    const encoded: number[] = [];
    for (const piece of pieces) {
      const bytes = encodeBytes(piece);
      if (bytes.length < 2) {
        encoded.push(...bytes);
        continue;
      }
      encoded.push(...this.mergePiece(bytes));
    }
    return encoded;
  }

  /**
   * Encode text into token ids, replacing any special token occurrence with a
   * single reserved id. Special tokens may occur anywhere in the text.
   */
  encode(text: string): number[] {
    const normalized = normalizeText(text);
    if (this.specialRegex === null) {
      return this.encodeOrdinary(normalized);
    }

    const ids: number[] = [];
    let last = 0;
    this.specialRegex.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = this.specialRegex.exec(normalized)) !== null) {
      if (match.index > last) {
        ids.push(...this.encodeOrdinary(normalized.slice(last, match.index)));
      }
      const id = this.specialId(match[0]);
      if (id >= this.specialBase) {
        ids.push(id);
      } else {
        ids.push(...this.encodeOrdinary(match[0]));
      }
      last = match.index + match[0].length;
    }
    if (last < normalized.length) {
      ids.push(...this.encodeOrdinary(normalized.slice(last)));
    }
    return ids;
  }

  /** Decode a sequence of token ids back into text. */
  decode(ids: Iterable<number>): string {
    const base = this.specialBase;
    const replacement = new TextEncoder().encode(REPLACEMENT_CHAR);
    const chunks: Uint8Array[] = [];
    let total = 0;
    for (const id of ids) {
      let bytes: Uint8Array;
      if (id >= base) {
        const index = id - base;
        const token = this.specialTokens[index];
        bytes = token !== undefined ? new TextEncoder().encode(token) : replacement;
      } else {
        bytes = this.vocabularyBytes[id] ?? replacement;
      }
      chunks.push(bytes);
      total += bytes.length;
    }
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return decodeBytes(out);
  }

  /** Export the learned vocabulary so it can be persisted. */
  toVocab(): BPEVocab {
    const merges: Array<[number, number]> = [];
    for (const [key, id] of this.merges) {
      merges.push([key, id]);
    }
    merges.sort((x, y) => x[1] - y[1]);
    return { merges, specialTokens: [...this.specialTokens] };
  }

  /** Rebuild a tokenizer instance from a persisted vocabulary. */
  static fromVocab(vocab: BPEVocab): BPETokenizer {
    for (const token of vocab.specialTokens) {
      validateSpecialToken(token);
    }
    const merges = new Map<number, number>();
    const mergePairs: Array<[number, number]> = [];
    for (const [key, id] of vocab.merges) {
      merges.set(key, id);
      mergePairs.push([
        Math.floor(key / PAIR_MULTIPLIER),
        key % PAIR_MULTIPLIER,
      ]);
    }
    // Order matters for rebuilding the byte table: sort by assigned id.
    const orderedPairs = [...mergePairs].sort((x, y) => {
      return (
        merges.get(pairKey(x[0], x[1]))! - merges.get(pairKey(y[0], y[1]))!
      );
    });
    const tokenizer = new BPETokenizer(vocab.specialTokens);
    tokenizer.merges = merges;
    tokenizer.vocabularyBytes = buildVocabulary(merges.size, orderedPairs);
    return tokenizer;
  }

  // --- internals -------------------------------------------------------------

  /**
   * Greedy rank-aware merge of a single byte piece: repeatedly merge the
   * adjacent pair that was trained earliest.
   */
  private mergePiece(bytes: number[]): number[] {
    let ids = bytes.slice();
    for (;;) {
      let bestPosition = -1;
      let bestKey = -1;
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i + 1 < ids.length; i++) {
        const candidate = pairKey(ids[i], ids[i + 1]);
        const id = this.merges.get(candidate);
        if (id !== undefined) {
          const rank = id - BASE_VOCABULARY;
          if (rank < bestRank) {
            bestRank = rank;
            bestKey = candidate;
            bestPosition = i;
          }
        }
      }
      if (bestPosition === -1) break;
      const mergedId = this.merges.get(bestKey)!;
      ids = [
        ...ids.slice(0, bestPosition),
        mergedId,
        ...ids.slice(bestPosition + 2),
      ];
    }
    return ids;
  }
}

/** Build the id -> bytes table used for decoding. */
function buildVocabulary(
  mergesMade: number,
  mergePairs: Array<[number, number]>,
): Uint8Array[] {
  const vocabulary: Uint8Array[] = new Array(mergesMade);
  for (let b = 0; b < BASE_VOCABULARY; b++) {
    vocabulary[b] = Uint8Array.of(b);
  }
  for (let i = 0; i < mergesMade; i++) {
    const [a, b] = mergePairs[i];
    const left = vocabulary[a];
    const right = vocabulary[b];
    const merged = new Uint8Array(left.length + right.length);
    merged.set(left, 0);
    merged.set(right, left.length);
    vocabulary[BASE_VOCABULARY + i] = merged;
  }
  return vocabulary;
}