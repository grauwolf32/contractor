import configIdentityCases from "../../../api/testdata/v1alpha1/config-identity-cases.json";
import { describe, expect, it } from "vitest";

import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
  isConfigSelector,
} from "./workflows";

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

describe("configuration selector grammar", () => {
  const { id, version } = configIdentityCases;

  it("accepts every shared valid name and version pair", () => {
    for (const name of [...id.valid, "a".repeat(id.maxLength)])
      for (const value of [...version.valid, "1".repeat(version.maxLength)])
        expect(isConfigSelector(`${name}@${value}`), `${name}@${value}`).toBe(
          true,
        );
  });

  it("rejects a shared invalid name or version and any other shape", () => {
    for (const name of [...id.invalid, "a".repeat(id.maxLength + 1)])
      expect(isConfigSelector(`${name}@1`), name).toBe(false);
    for (const value of [...version.invalid, "1".repeat(version.maxLength + 1)])
      expect(isConfigSelector(`a@${value}`), value).toBe(false);
    for (const value of ["", "a", "@1", "a@", "a@1@2", "a@@1"])
      expect(isConfigSelector(value), value).toBe(false);
  });
});
