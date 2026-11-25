import { fileURLToPath } from "node:url";

import { checkBrowserSpecs, runBrowserGate } from "./browser-gate.mjs";

// Every self-contained API-mocked spec belongs here. The remaining spec files
// run against a real process stack in tests/ui-stack or tests/e2e.
const mockedSpecs = [
  "e2e/llm-cost.spec.ts",
  "e2e/archive-preview.spec.ts",
  "e2e/artifact-preview.spec.ts",
  "e2e/material-detail-design.spec.ts",
  "e2e/sitemap-preview.spec.ts",
  "e2e/engagement-preview.spec.ts",
  "e2e/new-check-design.spec.ts",
  "e2e/project-material-names.spec.ts",
  "e2e/audit-labels.spec.ts",
  "e2e/audit-hypotheses.spec.ts",
  "e2e/audit-presets.spec.ts",
  "e2e/audit-research.spec.ts",
  "e2e/audit-workspace.spec.ts",
  "e2e/catalog.spec.ts",
  "e2e/check-events.spec.ts",
  "e2e/decision-focus.spec.ts",
  "e2e/dialogs.spec.ts",
  "e2e/evals-comparison.spec.ts",
  "e2e/evals-design.spec.ts",
  "e2e/evals-setup.spec.ts",
  "e2e/evals-skills.spec.ts",
  "e2e/project-finding-sources.spec.ts",
  "e2e/project-facts.spec.ts",
  "e2e/issue-review-design.spec.ts",
  "e2e/screenshot-polish.spec.ts",
  "e2e/git-artifacts.spec.ts",
  "e2e/model-selection.spec.ts",
  "e2e/operations-forms.spec.ts",
  "e2e/runtime-slots.spec.ts",
  "e2e/owner-lists.spec.ts",
  "e2e/responsive-layout.spec.ts",
  "e2e/run-drafts.spec.ts",
  "e2e/runs-navigation.spec.ts",
  "e2e/scheduler-settings.spec.ts",
  "e2e/studio.spec.ts",
  "e2e/web-scans.spec.ts",
];

const processSpecs = [
  "e2e/scanner-findings.spec.ts",
  "e2e/web-engagement.spec.ts",
  "e2e/audits.spec.ts",
  "e2e/evals-stack.spec.ts",
  "e2e/lifecycle-controls.spec.ts",
  "e2e/performance.spec.ts",
  "e2e/project-workspace.spec.ts",
  "e2e/run-repeat.spec.ts",
  "e2e/scan-tools.spec.ts",
  "e2e/stack.spec.ts",
  "e2e/studio-catalog.spec.ts",
];
await checkBrowserSpecs(fileURLToPath(new URL("../e2e", import.meta.url)), [
  ...mockedSpecs,
  ...processSpecs,
]);
await runBrowserGate("mocked-browser", "Mocked browser", mockedSpecs);
