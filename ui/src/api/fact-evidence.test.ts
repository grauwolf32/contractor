import { describe, expect, it } from "vitest";
import fixture from "../../../api/testdata/finding-fact-evidence-cases.json";
import { conformsTo } from "./contract";

describe("shared fact evidence projection", () => {
  for (const entry of fixture.publicDocuments) {
    it(entry.name, () => {
      expect(
        conformsTo(
          entry.schema as Parameters<typeof conformsTo>[0],
          entry.document,
        ),
      ).toBe(entry.valid);
    });
  }
});
