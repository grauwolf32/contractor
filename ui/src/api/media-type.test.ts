import mediaTypeCases from "../../../api/testdata/v1alpha1/media-type-cases.json";
import { describe, expect, it } from "vitest";

import { MEDIA_TYPE_PATTERN } from "./artifacts";

describe("media type grammar", () => {
  it.each(mediaTypeCases.valid)("accepts shared valid case %j", (value) => {
    expect(MEDIA_TYPE_PATTERN.test(value)).toBe(true);
  });

  it.each(mediaTypeCases.invalid)("rejects shared invalid case %j", (value) => {
    expect(MEDIA_TYPE_PATTERN.test(value)).toBe(false);
  });
});
