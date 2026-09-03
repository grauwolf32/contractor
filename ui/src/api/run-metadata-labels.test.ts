import { describe, expect, it } from "vitest";

import { safeRunMetadataLabels } from "./run-metadata-labels";

describe("Run metadata labels", () => {
  it("returns one detached deterministic string map", () => {
    const source = { purpose: "eval", "eval.id": "eval_01" };
    const result = safeRunMetadataLabels(source);
    expect(result).toEqual({ "eval.id": "eval_01", purpose: "eval" });
    source.purpose = "changed";
    expect(result.purpose).toBe("eval");
  });

  it.each([
    null,
    [],
    { Upper: "value" },
    { "contractor.internal": "value" },
    { purpose: "" },
    { purpose: "Ж".repeat(129) },
  ])("rejects an invalid response map %#", (labels) => {
    expect(() => safeRunMetadataLabels(labels)).toThrow(TypeError);
  });
});
