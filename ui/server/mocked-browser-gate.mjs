import { readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { runBrowserGate } from "./browser-gate.mjs";

// Every self-contained API-mocked spec belongs here. The remaining spec files
// run against a real process stack in tests/ui-stack or tests/e2e.
const mockedSpecs = [
  "e2e/archive-preview.spec.ts",
  "e2e/artifact-preview.spec.ts",
  "e2e/audit-presets.spec.ts",
  "e2e/catalog.spec.ts",
  "e2e/check-events.spec.ts",
  "e2e/dialogs.spec.ts",
  "e2e/evals-comparison.spec.ts",
  "e2e/evals-design.spec.ts",
  "e2e/evals-setup.spec.ts",
  "e2e/evals-skills.spec.ts",
  "e2e/git-artifacts.spec.ts",
  "e2e/operations-forms.spec.ts",
  "e2e/owner-lists.spec.ts",
  "e2e/responsive-layout.spec.ts",
  "e2e/run-drafts.spec.ts",
  "e2e/runs-navigation.spec.ts",
];

const processSpecs = [
  "e2e/audits.spec.ts",
  "e2e/evals-stack.spec.ts",
  "e2e/lifecycle-controls.spec.ts",
  "e2e/performance.spec.ts",
  "e2e/project-workspace.spec.ts",
  "e2e/run-repeat.spec.ts",
  "e2e/scan-tools.spec.ts",
  "e2e/scheduler-settings.spec.ts",
  "e2e/stack.spec.ts",
];
const e2eDir = fileURLToPath(new URL("../e2e", import.meta.url));
const listedSpecs = new Set([...mockedSpecs, ...processSpecs]);
const actualSpecs = new Set(
  (await readdir(e2eDir))
    .filter((name) => name.endsWith(".spec.ts"))
    .map((name) => `e2e/${name}`),
);
const unlisted = [...actualSpecs].filter((name) => !listedSpecs.has(name));
const missing = [...listedSpecs].filter((name) => !actualSpecs.has(name));
if (unlisted.length !== 0 || missing.length !== 0) {
  throw new Error(
    `browser spec inventory differs: unlisted=${unlisted.join(", ")} missing=${missing.join(", ")}`,
  );
}

await runBrowserGate("Mocked browser", mockedSpecs);
