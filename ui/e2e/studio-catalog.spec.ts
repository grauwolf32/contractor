import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

import { readDraft } from "../src/routes/studio/document";
import { buildGraph } from "../src/routes/studio/graph";
import { validateDraft } from "../src/routes/studio/validation";

// The shipped catalog: Studio loads every authored definition in configs/
// without inventing local errors. Only tests/ui-stack runs this spec (it may
// read configs/); unit tests use fixture copies in src/routes/studio/testdata.
const configRoot = resolve(process.cwd(), "../configs");
type Kind = ReturnType<typeof readDraft>["kind"];
const DIRECTORIES: readonly (readonly [string, Kind])[] = [
  ["workflows", "Workflow"],
  ["audit-profiles", "AuditProfile"],
  ["agent-templates", "AgentTemplate"],
];

for (const [directory, kind] of DIRECTORIES) {
  const files = readdirSync(resolve(configRoot, directory)).filter((file) =>
    file.endsWith(".yaml"),
  );
  test(`Studio lists shipped ${directory}`, () => {
    expect(files.length).toBeGreaterThan(0);
  });
  for (const file of files) {
    test(`Studio loads shipped ${directory}/${file} without local errors`, () => {
      const source = readFileSync(resolve(configRoot, directory, file), "utf8");
      const draft = readDraft(source);
      expect(draft.kind).toBe(kind);
      expect(draft.source).toBe(source);
      expect(
        validateDraft(draft).filter((problem) => problem.severity === "error"),
      ).toEqual([]);
      expect(buildGraph(draft).nodes.length).toBeGreaterThan(0);
    });
  }
}
