import { build } from "esbuild";

await build({
  entryPoints: ["src/tokenizer/index.ts"],
  bundle: true,
  outfile: "public/tokenizer-bundle.js",
  format: "esm",
  target: "es2020",
  platform: "browser",
  external: [],
  define: { "process.env.NODE_ENV": '"production"' },
  minify: true,
  sourcemap: true,
});

console.log("Built public/tokenizer-bundle.js");