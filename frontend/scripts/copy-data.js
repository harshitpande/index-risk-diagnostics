// Copies the dashboard data contract (data/dashboard/*.json, written by pipeline/export_json.py)
// into frontend/public/ so Vite serves it at /snapshot.json etc. Runs as predev/prebuild.
// Paths resolve from this file's location, not the cwd, so it works both from frontend/ and via
// `npm --prefix frontend` from the repo root (the Vercel build). Exits non-zero on a missing or
// unparseable file: a failed build keeps the last good deploy live instead of shipping bad data.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.resolve(here, "..", "..", "data", "dashboard");
const destDir = path.resolve(here, "..", "public");
const FILES = ["snapshot.json", "timeseries.json", "montecarlo.json"];

fs.mkdirSync(destDir, { recursive: true });

for (const name of FILES) {
  const src = path.join(srcDir, name);
  let text;
  try {
    text = fs.readFileSync(src, "utf8");
  } catch (err) {
    console.error(`copy-data: cannot read ${src} (${err.code ?? err.message})`);
    process.exit(1);
  }
  try {
    JSON.parse(text);
  } catch (err) {
    console.error(`copy-data: ${src} is not valid JSON (${err.message})`);
    process.exit(1);
  }
  fs.copyFileSync(src, path.join(destDir, name));
  console.log(`copy-data: copied ${name} (${(Buffer.byteLength(text) / 1024).toFixed(1)} KB)`);
}
