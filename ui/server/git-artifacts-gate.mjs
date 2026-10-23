import { runBrowserGate } from "./browser-gate.mjs";

await runBrowserGate("git-artifacts", "Git artifacts", [
  "e2e/git-artifacts.spec.ts",
]);
