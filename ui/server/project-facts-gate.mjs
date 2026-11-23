import { runBrowserGate } from "./browser-gate.mjs";

await runBrowserGate("project-facts", "Project facts", [
  "e2e/project-facts.spec.ts",
  "e2e/project-finding-sources.spec.ts",
  "e2e/audit-hypotheses.spec.ts",
]);
