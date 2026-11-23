import { describe, expect, it } from "vitest";
import cases from "../../../../../api/testdata/nuclei-input-cases.json";
import { rulepackMetadataError } from "./rulepack-metadata";

describe("shared Nuclei rulepack metadata contract", () => {
  it.each(cases.rulepackMetadataCases)(
    "$name",
    ({ mediaType, size, valid }) => {
      expect(rulepackMetadataError(mediaType, size) === undefined).toBe(valid);
    },
  );
});
