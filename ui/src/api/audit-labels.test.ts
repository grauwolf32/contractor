import { describe, expect, it } from "vitest";
import fixture from "../../../api/testdata/v1alpha1/placement-label-cases.json";
import { auditLabels, labelPlacementError } from "./audit-labels";

describe("Audit labels", () => {
  it.each(fixture.metadataCases)(
    "shares placement extraction: $name",
    ({ labels, valid }) => {
      expect(
        labelPlacementError(labels as Record<string, string>) === undefined,
      ).toBe(valid);
    },
  );
  it("keeps ordinary labels and rejects provenance collisions and duplicates", () => {
    expect(
      auditLabels([
        { key: "cloud", value: "aws" },
        { key: "team", value: "appsec" },
        { key: "debug", value: "" },
      ]),
    ).toEqual({
      labels: { cloud: "aws", team: "appsec", debug: "" },
      error: undefined,
    });
    expect(
      auditLabels([{ key: "audit.role", value: "check" }]).error,
    ).toBeDefined();
    expect(
      auditLabels([
        { key: "team", value: "a" },
        { key: "team", value: "b" },
      ]).error,
    ).toBeDefined();
    expect(auditLabels([{ key: "", value: "" }]).error).toBeDefined();
    expect(
      auditLabels([{ key: "team:invalid", value: "" }]).error,
    ).toBeDefined();
    expect(
      auditLabels(
        Array.from({ length: fixture.maxAuditLabels }, (_, index) => ({
          key: `tag${index}`,
          value: "value",
        })),
      ).error,
    ).toBeUndefined();
    expect(
      auditLabels(
        Array.from({ length: fixture.maxAuditLabels + 1 }, (_, index) => ({
          key: `tag${index}`,
          value: "value",
        })),
      ).error,
    ).toBeDefined();
  });
});
