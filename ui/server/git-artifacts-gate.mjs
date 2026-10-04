import { runBrowserGate } from "./browser-gate.mjs";

await runBrowserGate("Git artifacts", ["e2e/git-artifacts.spec.ts"]);
