import configIdentityCases from "../../../api/testdata/v1alpha1/config-identity-cases.json";
import { describe, expect, it } from "vitest";

import { CONFIG_NAME_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

describe("configuration identity grammar", () => {
  const { id, version } = configIdentityCases;

  it.each(id.valid)("accepts shared valid name %j", (value) => {
    expect(CONFIG_NAME_PATTERN.test(value)).toBe(true);
  });

  it.each(id.invalid)("rejects shared invalid name %j", (value) => {
    expect(CONFIG_NAME_PATTERN.test(value)).toBe(false);
  });

  it.each(version.valid)("accepts shared valid version %j", (value) => {
    expect(CONFIG_VERSION_PATTERN.test(value)).toBe(true);
  });

  it.each(version.invalid)("rejects shared invalid version %j", (value) => {
    expect(CONFIG_VERSION_PATTERN.test(value)).toBe(false);
  });

  it("applies the shared length bounds", () => {
    expect(CONFIG_NAME_PATTERN.test("a".repeat(id.maxLength))).toBe(true);
    expect(CONFIG_NAME_PATTERN.test("a".repeat(id.maxLength + 1))).toBe(false);
    expect(CONFIG_VERSION_PATTERN.test("1".repeat(version.maxLength))).toBe(
      true,
    );
    expect(CONFIG_VERSION_PATTERN.test("1".repeat(version.maxLength + 1))).toBe(
      false,
    );
  });
});
