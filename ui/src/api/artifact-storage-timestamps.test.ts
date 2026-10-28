import cases from "../../../api/testdata/public/artifact-storage-timestamps.json";
import { describe, expect, it } from "vitest";
import { schemaViolation } from "./contract";

describe("Shared Artifact storage timestamps", () => {
  for (const testCase of cases)
    it(testCase.name, () => {
      expect(
        schemaViolation("ArtifactMetadata", testCase.value) === undefined,
      ).toBe(testCase.valid);
    });
});
