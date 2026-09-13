import { performance } from "node:perf_hooks";
import { BPETokenizer, BASE_VOCABULARY } from "../src/tokenizer/index.js";

function makeCorpus(words: number): string {
  const pool = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "space",
    "model",
    "learning",
    "transformer",
    "attention",
    "language",
    "token",
    "byte",
    "statistics",
    "machine",
  ];
  const parts: string[] = [];
  for (let i = 0; i < words; i++) {
    parts.push(pool[i % pool.length]);
  }
  return parts.join(" ") + ".\n";
}

async function main(): Promise<void> {
  const corpus = makeCorpus(50_000);
  console.log(`corpus: ${corpus.length.toLocaleString()} bytes`);

  const tokenizer = BPETokenizer.createWithSpecialTokens(["<|endoftext|>"]);
  const vocabSize = 4_000;

  const start = performance.now();
  await tokenizer.train(corpus, {
    vocabularySize: vocabSize,
    onProgress: (p) => console.log(`  merge #${p.mergeIndex + 1} ...`),
  });
  const elapsed = performance.now() - start;
  console.log(
    `trained in ${(elapsed / 1000).toFixed(2)}s, vocab=${tokenizer.vocabularySize}`,
  );

  const sample = "the quick brown fox jumps over the lazy dog.";
  const ids = tokenizer.encode(sample);
  console.log(`\nencode: "${sample}"`);
  console.log(`  ids   : [${ids.join(", ")}]`);
  console.log(`  decode: "${tokenizer.decode(ids)}"`);

  const eot = tokenizer.encode("<|endoftext|>")[0];
  console.log(`\n<|endoftext|> -> id ${eot}`);

  const coverage = 1 - ids.length / new TextEncoder().encode(sample).length;
  console.log(`compression: ${(coverage * 100).toFixed(1)}% fewer tokens`);
  console.log(`base tokens below ${BASE_VOCABULARY}: ${ids.filter((i) => i < BASE_VOCABULARY).length}`);

  const restored = BPETokenizer.fromVocab(tokenizer.toVocab());
  console.log(
    "\nserialization round-trip:",
    JSON.stringify(restored.encode(sample)) === JSON.stringify(ids) ? "ok" : "FAILED",
  );
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});