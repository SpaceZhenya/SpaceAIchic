/**
 * Build script: bundles src/ui/page.ts into a single self-contained index.html
 * (via esbuild iife → injected into template.html). No external script tags,
 * so GitHub Pages serves it without a build step.
 *
 * Usage: npm run build   (or node scripts/build-page.mjs)
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const entry  = resolve(root, "src/ui/page.ts");
const template = resolve(root, "template.html");
const out = resolve(root, "index.html");

const result = await build({
  entryPoints: [entry],
  bundle: true,
  format: "iife",
  minify: true,
  target: "es2020",
  outfile: "-",       // we capture stdout via write plugin, but api 'write' is false so result is returned
  write: false,
  define: { "process.env.NODE_ENV": '"production"' },
  drop: ["debugger"],
});

const bundle = result.outputFiles[0].text;
const html = readFileSync(template, "utf8");

if (!html.includes("<!--APP-->")) {
  throw new Error("template.html does not contain <!--APP--> placeholder");
}

const finalHtml = html.replace("<!--APP-->", `<script>\n${bundle}\n</script>`);

writeFileSync(out, finalHtml, "utf8");

const sizeKb = (Buffer.byteLength(finalHtml) / 1024).toFixed(1);
console.log(`✅  Built index.html (${sizeKb} KB, single self-contained file)`);
console.log(`   Contains: GPT forward/backward, AdamW, BPE tokenizer, training loop, UI — all in one file.`);

// Sanity: no external script tags
if (/<script\s+src=/i.test(finalHtml)) {
  throw new Error("output contains <script src= — expected only inline scripts");
}
console.log(`   Verified: no external script tags.`);