import { describe, expect, it } from "vitest";
import cases from "../../../api/testdata/public/project-active-check-policy.json";
import { requireActiveChecksPolicy } from "./projects";

describe("shared Project active checks policy", () => {
  for (const testCase of cases) {
    it(JSON.stringify(testCase.value), () => {
      if (testCase.valid)
        expect(requireActiveChecksPolicy(testCase.value)).toBe(testCase.value);
      else
        expect(() => requireActiveChecksPolicy(testCase.value)).toThrow(
          TypeError,
        );
    });
  }
});
