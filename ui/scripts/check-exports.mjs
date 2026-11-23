// make lint: fails on exports of production modules under src/ that no other
// production module imports (export-gate.mjs), modelled on
// tools/testonlyexports. scripts/exports-baseline.txt lists the ones lint
// tolerates for now, one "<file>:<export> <task ID>" line each, where the
// task is an open task file in ../tasks that resolves them. The check fails
// on a finding the baseline does not list, on an entry that no longer
// matches one, and on an entry whose task is finished or missing, so the
// baseline only shrinks. Fixing one export can expose another that only it
// used; the next run reports that one.
//
// Usage: node scripts/check-exports.mjs [--write-baseline TASK] [UI_DIR]
// --write-baseline rewrites the file with every current finding under TASK;
// use it only to drop entries, and review that its diff only removes lines.
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { analyzeExports } from "./export-gate.mjs";

const args = process.argv.slice(2);
const writeIndex = args.indexOf("--write-baseline");
const writeTask = writeIndex < 0 ? undefined : args[writeIndex + 1];
const rest = args.filter(
  (_arg, index) => index !== writeIndex && index !== writeIndex + 1,
);
const ui = resolve(rest[0] ?? new URL("..", import.meta.url).pathname);
const baselinePath = resolve(ui, "scripts/exports-baseline.txt");
const tasksDir = resolve(ui, "../tasks");

const { findings, errors } = analyzeExports(ui);
const keyOf = (finding) => finding.split(" ")[0];

if (writeTask !== undefined) {
  const header = readFileSync(baselinePath, "utf8")
    .split("\n")
    .filter((line) => line.startsWith("#"));
  writeFileSync(
    baselinePath,
    [
      ...header,
      "",
      ...findings.map((f) => `${keyOf(f)} ${writeTask}`),
      "",
    ].join("\n"),
  );
  console.log(`wrote ${findings.length} entries to ${baselinePath}`);
  process.exit(0);
}

function openTask(id) {
  const prefix = `${id.toLowerCase()}-`;
  return (
    existsSync(tasksDir) &&
    readdirSync(tasksDir).some(
      (name) => name.startsWith(prefix) && name.endsWith(".yml"),
    )
  );
}

const baseline = new Map();
for (const line of readFileSync(baselinePath, "utf8").split("\n")) {
  const trimmed = line.trim();
  if (trimmed === "" || trimmed.startsWith("#")) continue;
  const [key, task] = trimmed.split(/\s+/);
  baseline.set(key, task);
}

const problems = [...errors];
const current = new Set();
for (const finding of findings) {
  const key = keyOf(finding);
  current.add(key);
  if (!baseline.has(key))
    problems.push(
      `${finding}: no other production module imports it; drop the export, import it, or test through the module's production API`,
    );
}
for (const [key, task] of baseline) {
  if (!current.has(key))
    problems.push(
      `scripts/exports-baseline.txt: ${key} no longer needs an entry; delete the line`,
    );
  else if (task === undefined || !openTask(task))
    problems.push(
      `scripts/exports-baseline.txt: ${key} names ${task ?? "no task"}, which is not an open task in tasks/`,
    );
}
for (const problem of problems) console.error(problem);
if (problems.length > 0) {
  console.error(`${problems.length} export finding(s).`);
  process.exit(1);
}
console.log(`UI exports: ok (${baseline.size} tolerated in the baseline)`);
