import cases from "../../../api/testdata/public/audit-material-cases.json";
import { describe, expect, it } from "vitest";
import { schemaViolation } from "./contract";
import type { PublicSchemaName } from "./generated/response-schemas";

describe("Shared Audit material contracts", () => {
  for (const testCase of cases)
    it(testCase.name, () => {
      expect(
        schemaViolation(testCase.schema as PublicSchemaName, testCase.value) ===
          undefined,
      ).toBe(testCase.valid);
    });
});
