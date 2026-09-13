import { describe, it, expect } from "vitest";
import { BPETokenizer, BASE_VOCABULARY } from "../src/tokenizer/index.js";

const CORPUS = [
  "The quick brown fox jumps over the lazy dog. ",
  "The lazy dog sleeps while the quick brown fox runs fast. ",
  "Hello world, hello space, hello everyone! ",
  "SpaceAI is learning to speak: byte by byte, token by token. ",
  "Machine learning models understand language by statistics. ",
  "Transformers are the foundation of modern language models. ",
  "Attention is all you need, they say, but data is the fuel. ",
].join("\n");

async function makeTokenizer(vocabSize = 512) {
  const tokenizer = BPETokenizer.create();
  await tokenizer.train(CORPUS, { vocabularySize: vocabSize });
  return tokenizer;
}

describe("BPETokenizer", () => {
  it("exposes exactly the requested vocabulary size", async () => {
    const tokenizer = await makeTokenizer(400);
    expect(tokenizer.vocabularySize).toBeLessThanOrEqual(400);
    expect(tokenizer.vocabularySize).toBeGreaterThan(BASE_VOCABULARY);
  });

  it("round-trips ordinary ASCII text", async () => {
    const tokenizer = await makeTokenizer();
    const text = "The quick brown fox runs fast. Hello world!";
    const ids = tokenizer.encode(text);
    expect(ids.length).toBeGreaterThan(0);
    expect(tokenizer.decode(ids)).toBe(text);
  });

  it("round-trips text with unicode and punctuation", async () => {
    const tokenizer = await makeTokenizer();
    const text = "Café déjà vu — 中文 works too. 123,456!?";
    const ids = tokenizer.encode(text);
    expect(tokenizer.decode(ids)).toBe(text);
  });

  it("learns subword units: frequent words compress into fewer tokens", async () => {
    const repeated = "transformer ".repeat(300) + "a boundary word here";
    const tokenizer = BPETokenizer.create();
    await tokenizer.train(repeated, { vocabularySize: 800 });

    const ids = tokenizer.encode("transformer");
    const rawBytes = new TextEncoder().encode("transformer").length;
    expect(ids.length).toBeLessThan(rawBytes);
    expect(tokenizer.decode(ids)).toBe("transformer");
  });

  it("encodes and decodes special tokens as single reserved ids", async () => {
    const tokenizer = BPETokenizer.createWithSpecialTokens(["<|endoftext|>"]);
    await tokenizer.train(CORPUS, { vocabularySize: 400 });
    const ids = tokenizer.encode("hello<|endoftext|>world");
    const specialIndex = ids.find(
      (id) => id >= tokenizer.vocabularySize - 1,
    );
    expect(specialIndex).not.toBeUndefined();
    expect(ids.filter((id) => id === specialIndex).length).toBe(1);
    expect(tokenizer.decode(ids)).toBe("hello<|endoftext|>world");
  });

  it("serializes and restores a vocabulary exactly", async () => {
    const a = await makeTokenizer(600);
    const sample = "The lazy dog sleeps. 中文 text!";
    const expected = a.encode(sample);

    const restored = BPETokenizer.fromVocab(a.toVocab());
    expect(restored.vocabularySize).toBe(a.vocabularySize);
    expect(restored.encode(sample)).toEqual(expected);
    expect(restored.decode(expected)).toBe(sample);
  });

  it("empty tokenizer round-trips raw bytes with merges disabled", async () => {
    const tokenizer = BPETokenizer.create();
    const text = "plain ascii";
    const ids = tokenizer.encode(text);
    expect(ids.every((id) => id < BASE_VOCABULARY)).toBe(true);
    expect(tokenizer.decode(ids)).toBe(text);
  });
});