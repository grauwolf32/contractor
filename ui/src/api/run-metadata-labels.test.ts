import labelCases from "../../../api/testdata/v1alpha1/run-metadata-label-cases.json";
import { describe, expect, it } from "vitest";

import {
  normalizeRunMetadataLabelSelectors,
  RUN_METADATA_LABEL_LIMIT,
  runMetadataLabelKeyError,
  runMetadataLabelValueError,
} from "./run-metadata-labels";

/** Whether a label map passes the authoring rules a Run form applies. */
function authorable(value: unknown): boolean {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const entries = Object.entries(value);
  return (
    entries.length <= RUN_METADATA_LABEL_LIMIT &&
    entries.every(
      ([key, entry]) =>
        runMetadataLabelKeyError(key) === undefined &&
        runMetadataLabelValueError(entry) === undefined,
    )
  );
}

describe("Run metadata labels", () => {
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

describe("shared run metadata label cases", () => {
  it.each(labelCases.valid)("accepts $name like Go and Python", ({ value }) => {
    expect(authorable(value)).toBe(true);
  });

  it.each(labelCases.invalid)(
    "rejects $name like Go and Python",
    (labelCase) => {
      expect(
        authorable("omit" in labelCase ? undefined : labelCase.value),
      ).toBe(false);
    },
  );
});
