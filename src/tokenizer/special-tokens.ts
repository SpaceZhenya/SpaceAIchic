/**
 * Special tokens are reserved vocabulary entries (e.g. <|endoftext|>) that are
 * always encoded as a single whole token and never split by BPE merges.
 */

export const DEFAULT_SPECIAL_TOKENS: readonly string[] = ["<|endoftext|>"];

import { encodeBytes } from "./unicode.js";

/** Escape a string so it can be used verbatim inside a RegExp. */
export function escapeRegExp(part: string): string {
  return part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Reject reserved-looking tokens that would break the encode/decode mapping. */
export function validateSpecialToken(token: string): void {
  if (token.length === 0) {
    throw new Error("special token must not be empty");
  }
  if (/\s/.test(token)) {
    throw new Error(`special token must not contain whitespace: "${token}"`);
  }
  if ([...encodeBytes(token)].length === 0) {
    throw new Error(`special token must be non-empty after encoding: "${token}"`);
  }
}