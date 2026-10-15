// make lint: fails on stylesheet classes no production code names, on @media
// widths outside the documented breakpoints and on stale allowlist entries
// (css-gate.mjs). Usage: node scripts/check-css.mjs [UI_DIR]
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { checkStyles } from "./css-gate.mjs";

const ui = resolve(process.argv[2] ?? new URL("..", import.meta.url).pathname);
const allowlist = JSON.parse(
  readFileSync(resolve(ui, "scripts/css-allowlist.json"), "utf8"),
);
const findings = checkStyles({ ui, allowlist });
for (const finding of findings) console.error(finding);
if (findings.length > 0) {
  console.error(
    `${findings.length} CSS finding(s). Delete unused rules; a class built at run time or rendered by a library goes into scripts/css-allowlist.json with its source; breakpoints are listed in src/ui/README.md.`,
  );
  process.exit(1);
}
console.log("CSS classes and breakpoints: ok");
