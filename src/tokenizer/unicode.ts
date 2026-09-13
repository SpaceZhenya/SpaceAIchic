/**
 * Unicode utilities for the tokenizer.
 *
 * The BPE tokenizer works on UTF-8 bytes internally, so plain-JS strings are
 * normalized to NFC first and then pretokenized with the classic GPT-2
 * regular expression before being encoded as bytes (0-255).
 */

/** Normalize a string to NFC so equivalent composed/decomposed forms unify. */
export function normalizeText(text: string): string {
  return typeof text.normalize === "function" ? text.normalize("NFC") : text;
}

/**
 * The well-known GPT-2 pretokenization pattern (no possessive quantifiers, so
 * it is valid JS regex). Each produced piece is treated as an indivisible unit
 * for BPE: merges are never allowed to span piece boundaries during encoding.
 */
export const PRETOKEN_PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** Split text into pretoken pieces according to the GPT-2 pattern. */
export function pretokenize(text: string): string[] {
  return text.match(PRETOKEN_PATTERN) ?? [];
}

/** Encode a single piece into its raw UTF-8 bytes. */
export function encodeBytes(piece: string): number[] {
  if (piece.length === 0) return [];
  return Array.from(new TextEncoder().encode(piece));
}

/** Decode raw UTF-8 bytes (0-255 ids) back into a string. */
export function decodeBytes(bytes: Uint8Array): string {
  if (bytes.length === 0) return "";
  return new TextDecoder().decode(bytes);
}

/** Replacement applied for unknown/invalid token ids during decode. */
export const REPLACEMENT_CHAR = "\uFFFD";