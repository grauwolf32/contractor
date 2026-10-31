import { TEST_SHARED_CONFIG } from "../test/shared-config";
import labelCases from "../../../api/testdata/v1alpha1/run-metadata-label-cases.json";
import { describe, expect, it } from "vitest";

import {
  normalizeRunMetadataLabelSelectors,
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
    entries.length <= TEST_SHARED_CONFIG.limits.runMetadataMaxLabels &&
    entries.every(
      ([key, entry]) =>
        runMetadataLabelKeyError(key, TEST_SHARED_CONFIG.limits) ===
          undefined &&
        runMetadataLabelValueError(entry, TEST_SHARED_CONFIG.limits) ===
          undefined,
    )
  );
}

describe("Run metadata labels", () => {
  it("normalizes exact selectors and preserves contradictory values", () => {
    expect(
      normalizeRunMetadataLabelSelectors(
        [
          { key: "debug", value: "" },
          { key: "purpose", value: "eval" },
          { key: "eval.leg", value: "b" },
          { key: "purpose", value: "eval" },
          { key: "eval.leg", value: "a" },
        ],
        TEST_SHARED_CONFIG.limits,
      ),
    ).toEqual([
      { key: "debug", value: "" },
      { key: "eval.leg", value: "a" },
      { key: "eval.leg", value: "b" },
      { key: "purpose", value: "eval" },
    ]);
  });

  it("provides field-specific authoring failures without trimming opaque values", () => {
    expect(
      runMetadataLabelKeyError("contractor.trace", TEST_SHARED_CONFIG.limits),
    ).toBe("The contractor. prefix is reserved.");
    expect(
      runMetadataLabelKeyError("Eval.ID", TEST_SHARED_CONFIG.limits),
    ).toContain("lowercase ASCII");
    expect(
      runMetadataLabelValueError("", TEST_SHARED_CONFIG.limits),
    ).toBeUndefined();
    expect(
      runMetadataLabelValueError("\0hidden", TEST_SHARED_CONFIG.limits),
    ).toContain("U+0000");
    expect(
      runMetadataLabelValueError(" value ", TEST_SHARED_CONFIG.limits),
    ).toBeUndefined();
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

it("applies received label policy independently of compiled defaults", () => {
  const limits = {
    ...TEST_SHARED_CONFIG.limits,
    runMetadataMaxLabels: 1,
    runMetadataKeyMaxBytes: 4,
    runMetadataValueMaxBytes: 5,
  };
  expect(runMetadataLabelKeyError("team", limits)).toBeUndefined();
  expect(runMetadataLabelKeyError("teams", limits)).toContain("4 UTF-8 bytes");
  expect(runMetadataLabelValueError("ééx", limits)).toBeUndefined();
  expect(runMetadataLabelValueError("ééxx", limits)).toContain("5 UTF-8 bytes");
  expect(() =>
    normalizeRunMetadataLabelSelectors(
      [
        { key: "one", value: "" },
        { key: "two", value: "" },
      ],
      limits,
    ),
  ).toThrow("1 entries");
});
