import { describe, expect, it } from "vitest";
import fixture from "../../../api/testdata/finding-fact-cases.json";
import {
  findingNeedsReview,
  findingSeverity,
  type ProjectFindingSummary,
} from "./audits";

describe("shared fact presentation", () => {
  for (const row of fixture.views) {
    it(row.name, () => {
      const finding = row as unknown as ProjectFindingSummary;
      expect(findingSeverity(finding)).toBe(row.severity);
      expect(findingNeedsReview(finding)).toBe(row.needsReview);
    });
  }
});
