import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { isAlias, parseDocument, stringify, visit } from "yaml";

import { MAX_SOURCE_BYTES, readDraft } from "../src/routes/studio/document";
import { buildGraph } from "../src/routes/studio/graph";
import { validateDraft } from "../src/routes/studio/validation";

// The shipped catalog follows Studio's documented import contract: trusted
// fixture aliases are explicitly expanded before visual editing. Only
// tests/ui-stack runs this spec (it may read configs/); unit tests use fixture
// copies in src/routes/studio/testdata.
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
      expect(Buffer.byteLength(source, "utf8")).toBeLessThanOrEqual(
        MAX_SOURCE_BYTES,
      );
      const document = parseDocument(source, { uniqueKeys: true });
      expect(document.errors).toEqual([]);
      expect(document.warnings).toEqual([]);
      let nodes = 0;
      let aliased = false;
      visit(document, (_, node) => {
        nodes++;
        aliased ||= isAlias(node);
      });
      expect(nodes).toBeLessThanOrEqual(20_000);
      if (aliased)
        expect(() => readDraft(source)).toThrow(
          "Expand YAML aliases before importing into the visual editor.",
        );
      const value = document.toJS({ maxAliasCount: 100 });
      const importSource = aliased
        ? stringify(value, { aliasDuplicateObjects: false })
        : source;
      const draft = readDraft(importSource);
      expect(draft.kind).toBe(kind);
      expect(draft.source).toBe(importSource);
      expect(draft.value).toEqual(value);
      expect(
        validateDraft(draft).filter((problem) => problem.severity === "error"),
      ).toEqual([]);
      expect(buildGraph(draft).nodes.length).toBeGreaterThan(0);
    });
  }
}
