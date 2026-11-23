import { TEST_SHARED_CONFIG } from "../test/shared-config";
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
      auditLabels(
        [
          { key: "cloud", value: "aws" },
          { key: "team", value: "appsec" },
          { key: "debug", value: "" },
        ],
        TEST_SHARED_CONFIG.limits,
      ),
    ).toEqual({
      labels: { cloud: "aws", team: "appsec", debug: "" },
      error: undefined,
    });
    expect(
      auditLabels(
        [{ key: "audit.role", value: "check" }],
        TEST_SHARED_CONFIG.limits,
      ).error,
    ).toBeDefined();
    expect(
      auditLabels(
        [
          { key: "team", value: "a" },
          { key: "team", value: "b" },
        ],
        TEST_SHARED_CONFIG.limits,
      ).error,
    ).toBeDefined();
    expect(
      auditLabels([{ key: "", value: "" }], TEST_SHARED_CONFIG.limits).error,
    ).toBeDefined();
    expect(
      auditLabels(
        [{ key: "team:invalid", value: "" }],
        TEST_SHARED_CONFIG.limits,
      ).error,
    ).toBeDefined();
    expect(
      auditLabels(
        Array.from({ length: fixture.maxAuditLabels }, (_, index) => ({
          key: `tag${index}`,
          value: "value",
        })),
        TEST_SHARED_CONFIG.limits,
      ).error,
    ).toBeUndefined();
    expect(
      auditLabels(
        Array.from({ length: fixture.maxAuditLabels + 1 }, (_, index) => ({
          key: `tag${index}`,
          value: "value",
        })),
        TEST_SHARED_CONFIG.limits,
      ).error,
    ).toBeDefined();
  });
});

it("uses the received Audit label allowance", () => {
  const limits = { ...TEST_SHARED_CONFIG.limits, auditMaxLabels: 1 };
  expect(
    auditLabels([{ key: "team", value: "one" }], limits).error,
  ).toBeUndefined();
  expect(
    auditLabels(
      [
        { key: "team", value: "one" },
        { key: "other", value: "two" },
      ],
      limits,
    ).error,
  ).toContain("at most 1 labels");
});
