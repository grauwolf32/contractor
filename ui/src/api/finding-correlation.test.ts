import { describe, expect, it } from "vitest";

import fixture from "../../../api/testdata/public/finding-correlation-cases.json";
import { conformsTo } from "./contract";

describe("finding correlation wire contract", () => {
  for (const entry of fixture.projections) {
    it(entry.name, () => {
      expect(conformsTo("FindingCorrelation", entry.document)).toBe(
        entry.schemaValid,
      );
    });
  }
});
