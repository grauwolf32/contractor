import { expect, it } from "vitest";
import fixture from "../../../api/testdata/finding-http-evidence-cases.json";
import { conformsTo } from "./contract";

for (const entry of fixture.cases) {
  it(`shared HTTP projections: ${entry.name}`, () => {
    expect(conformsTo("FindingProposalDocument", entry.proposal)).toBe(
      entry.valid,
    );
  });
}
