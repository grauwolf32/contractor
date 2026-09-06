import { describe, expect, it } from "vitest";

import {
  normalizeRunMetadataLabelSelectors,
  runMetadataLabelKeyError,
  runMetadataLabelValueError,
  safeRunMetadataLabels,
} from "./run-metadata-labels";

describe("Run metadata labels", () => {
  it("returns one detached deterministic string map", () => {
    const source = { purpose: "eval", "eval.id": "eval_01", debug: "" };
    const result = safeRunMetadataLabels(source);
    expect(result).toEqual({
      debug: "",
      "eval.id": "eval_01",
      purpose: "eval",
    });
    source.purpose = "changed";
    expect(result.purpose).toBe("eval");
  });

  it.each([
    null,
    [],
    { Upper: "value" },
    { "contractor.internal": "value" },
    { purpose: null },
    { purpose: "Ж".repeat(129) },
  ])("rejects an invalid response map %#", (labels) => {
    expect(() => safeRunMetadataLabels(labels)).toThrow(TypeError);
  });

  it("normalizes exact selectors and preserves contradictory values", () => {
    expect(
      normalizeRunMetadataLabelSelectors([
        { key: "debug", value: "" },
        { key: "purpose", value: "eval" },
        { key: "eval.leg", value: "b" },
        { key: "purpose", value: "eval" },
        { key: "eval.leg", value: "a" },
      ]),
    ).toEqual([
      { key: "debug", value: "" },
      { key: "eval.leg", value: "a" },
      { key: "eval.leg", value: "b" },
      { key: "purpose", value: "eval" },
    ]);
  });

  it("provides field-specific authoring failures without trimming opaque values", () => {
    expect(runMetadataLabelKeyError("contractor.trace")).toBe(
      "The contractor. prefix is reserved.",
    );
    expect(runMetadataLabelKeyError("Eval.ID")).toContain("lowercase ASCII");
    expect(runMetadataLabelValueError("")).toBeUndefined();
    expect(runMetadataLabelValueError("\0hidden")).toContain("U+0000");
    expect(runMetadataLabelValueError(" value ")).toBeUndefined();
  });
});
