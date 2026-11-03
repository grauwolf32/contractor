import { describe, expect, it } from "vitest";
import {
  formatList,
  groupFamilies,
  groupRows,
  readinessOf,
  scopeFieldsUsed,
  scopeSummary,
} from "./check-types";
import { openapiYaml, profileFixture, sourceZip, trace } from "./test-fixtures";

describe("common check types", () => {
  it("selects the newest published version without old compatibility inference", () => {
    const families = groupFamilies([
      profileFixture("inspection", { version: "2" }),
      profileFixture("inspection", { version: "10" }),
      profileFixture("inspection", { version: "1" }),
    ]);
    expect(families[0]?.preferred.ref.version).toBe("10");
    expect(groupRows(families)).toHaveLength(1);
  });
  it("requires the selected input formats and preserves missing source information", () => {
    expect(
      readinessOf(trace, {
        materials: [sourceZip],
        hasLiveTarget: false,
        needsLiveTarget: false,
      }).state,
    ).toBe("missing");
    expect(
      readinessOf(trace, {
        materials: [sourceZip, openapiYaml],
        hasLiveTarget: false,
        needsLiveTarget: false,
      }).state,
    ).toBe("ready");
  });
  it("does not assign the same material to two required inputs", () => {
    const profile = profileFixture("two-sources", {
      inputs: {
        a: { required: true, media_types: ["application/zip"] },
        b: { required: true, media_types: ["application/zip"] },
      },
    });
    expect(
      readinessOf(profile, {
        materials: [sourceZip],
        hasLiveTarget: false,
        needsLiveTarget: false,
      }),
    ).toMatchObject({
      state: "missing",
      missing: [{ kind: "shared", count: 1 }],
    });
  });
  it("uses published task limits and operation policy", () => {
    expect(scopeSummary(trace).size).toContain("256");
    expect(scopeFieldsUsed(trace).size).toBe(0);
    expect(
      scopeFieldsUsed(
        profileFixture("http", { policy: { active_operations: "allowed" } }),
      ).has("target"),
    ).toBe(false);
    expect(formatList(["application/json", "application/yaml"])).toBe(
      "JSON or YAML",
    );
  });
});
