import { describe, expect, it } from "vitest";
import { suggestCheckTypes } from "./suggestions";
const everything = () => true;
describe("published common check type suggestions", () => {
  it("suggests nothing for blank or unsupported objectives", () => {
    expect(suggestCheckTypes(" ", everything)).toEqual({});
    expect(
      suggestCheckTypes("verify ASVS LEVEL 1", everything).suggestion,
    ).toBeUndefined();
  });
  it("matches whole API words and readiness", () => {
    expect(
      suggestCheckTypes("Rapid review of capital letters", everything)
        .suggestion,
    ).toBeUndefined();
    expect(
      suggestCheckTypes(
        "Trace API endpoints",
        (name) => name === "openapi-operation-observe",
      ).suggestion?.checkType,
    ).toBe("openapi-operation-observe");
    expect(
      suggestCheckTypes("Trace API endpoints", () => false).suggestion,
    ).toBeUndefined();
  });
  it("orders specific scanner work and offers another ready common program", () => {
    const result = suggestCheckTypes(
      "SQL injection in API endpoints",
      everything,
    );
    expect(result.suggestion?.checkType).toBe("openapi-sqlmap-scan");
    expect(result.alternative?.checkType).toBe("openapi-operation-trace");
  });
  it.each(["sqlmap", "nuclei", "endpoints"])(
    "explains only the matching rule and material formats: %s",
    (objective) => {
      expect(
        suggestCheckTypes(objective, everything).suggestion?.reason,
      ).toMatch(
        /^Your objective mentions .+ Your project has a material whose format matches each input it needs\.$/u,
      );
    },
  );
});
