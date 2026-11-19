// Does the browser model score like finrag? Embeds the 69 reference questions with
// transformers.js (the library the static page loads), scores them against the passages of
// site/chunks.json, and compares with site/reference.json (fastembed, finrag's embed()).
// Fails if any answer/refuse decision differs.   node check_browser.mjs site fp32
import { readFileSync, writeFileSync } from "node:fs";
import { pipeline } from "@huggingface/transformers";

const [site, dtype] = [process.argv[2], process.argv[3] ?? "fp32"];
const MODEL = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";
const REVISION = "2c4055b12046f11709e9df2c122e59ffbdc2f900";
const data = JSON.parse(readFileSync(`${site}/chunks.json`, "utf8"));
const ref = JSON.parse(readFileSync(`${site}/reference.json`, "utf8"));

const extractor = await pipeline("feature-extraction", MODEL, { dtype, revision: REVISION });
let maxDiff = 0, mismatches = [];
for (const r of ref) {
  const out = await extractor(r.q, { pooling: "mean", normalize: true });
  const q = Array.from(out.data);
  let best = -1;
  for (const c of data.chunks) {
    let s = 0;
    for (let i = 0; i < q.length; i++) s += q[i] * c.v[i];
    if (s > best) best = s;
  }
  maxDiff = Math.max(maxDiff, Math.abs(best - r.best));
  if ((best >= data.min_score) !== r.passes) mismatches.push({ q: r.q, browser: best, ref: r.best });
}
const report = { dtype, questions: ref.length, max_score_diff: maxDiff, decision_mismatches: mismatches };
writeFileSync(`${site}/../browser-check-${dtype}.json`, JSON.stringify(report, null, 1));
console.log(JSON.stringify(report));
process.exit(mismatches.length ? 1 : 0);
