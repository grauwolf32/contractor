import { describe, expect, it } from "vitest";

import fixture from "../../../api/testdata/public/finding-export-cases.json";
import { conformsTo, schemaLimit } from "./contract";

describe("shared Audit export contract", () => {
  for (const entry of fixture.cases) {
    it(entry.name, () => {
      expect(conformsTo("PublishFindingExportRequest", entry.request)).toBe(
        entry.valid,
      );
    });
  }
  it("uses the Server's shared selection and download bounds", () => {
    expect(
      schemaLimit(
        "PublishFindingExportRequest",
        "maxItems",
        "properties",
        "findings",
      ),
    ).toBe(fixture.limits.maxFindings);
    expect(schemaLimit("FindingExportContent", "maxLength")).toBe(
      fixture.limits.maxContentBytes,
    );
    expect(schemaLimit("FindingExportDiagnosticsContent", "maxLength")).toBe(
      fixture.limits.maxDiagnosticsBytes,
    );
  });
});
