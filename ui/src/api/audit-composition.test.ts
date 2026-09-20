import { describe, expect, it } from "vitest";
import cases from "../../../api/testdata/audit-composition/public-cases.json";
import {
  validAuditComposition,
  validInventory,
  validPreparation,
} from "./audit-composition";
import type { Audit } from "./audits";

describe("Audit composition public contracts", () => {
  for (const test of cases) {
    it(test.name, () => {
      const validate =
        test.schema === "AuditInventory" ? validInventory : validPreparation;
      expect(validate(test.value)).toBe(test.valid);
    });
  }

  it("requires a Round only in the rounds phase", () => {
    const preparation = cases.find(
      (test) => test.name === "pending preparation",
    )!.value;
    // Only composition fields are consumed here; the API parser independently
    // checks the base Audit identity and response fields.
    const preparing = {
      phase: "preparing",
      baseline: {},
      preparation,
    } as Audit;
    expect(validAuditComposition(preparing)).toBe(true);
    expect(
      validAuditComposition({ ...preparing, currentRoundId: "invented" }),
    ).toBe(false);
    expect(validAuditComposition({ ...preparing, phase: "rounds" })).toBe(
      false,
    );
    expect(validAuditComposition({ ...preparing, phase: "inventory" })).toBe(
      false,
    );
    expect(validPreparation(preparation, true)).toBe(false);
  });
});
