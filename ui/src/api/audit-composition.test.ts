import { describe, expect, it } from "vitest";

import cases from "../../../api/testdata/audit-composition/public-cases.json";
import { audit } from "../test/builders";
import { requireAuditComposition } from "./audit-composition";
import { conformsTo } from "./contract";
import type { components } from "./generated/public";

type Preparation = components["schemas"]["AuditPreparation"];

describe("Audit composition public contracts", () => {
  for (const test of cases) {
    it(test.name, () => {
      expect(
        conformsTo(
          test.schema as
            | "AuditInventory"
            | "AuditPreparation"
            | "AuditGeneratedInventory"
            | "AuditScanProvenance",
          test.value,
        ),
      ).toBe(test.valid);
    });
  }
});

describe("requireAuditComposition", () => {
  const preparation = (name: string) =>
    cases.find((test) => test.name === name)!.value as unknown as Preparation;
  const accepted = preparation("accepted preparation");
  const pending = preparation("pending preparation");
  const role = Object.values(accepted.roles)[0]!;
  const inPhase = (phase: "preparing" | "inventory", preparation: object) => {
    const value: Record<string, unknown> = {
      ...audit("project_1", "audit_1", "active"),
      phase,
      preparation,
      baseline: {},
    };
    delete value.currentRoundId;
    return value as never;
  };

  it("accepts consistent preparations", () => {
    expect(() =>
      requireAuditComposition(inPhase("preparing", pending), 200),
    ).not.toThrow();
    expect(() =>
      requireAuditComposition(inPhase("inventory", accepted), 200),
    ).not.toThrow();
  });

  it.each([
    ["an unaccepted role after preparation", inPhase("inventory", pending)],
    [
      "more attempts than allowed",
      inPhase("preparing", {
        roles: { extract: { ...role, attempts: role.maxAttempts + 1 } },
      }),
    ],
    [
      "an output of another Run",
      inPhase("preparing", {
        roles: {
          extract: {
            ...role,
            outputs: Object.fromEntries(
              Object.entries(role.outputs).map(([name, output]) => [
                name,
                { ...output, runId: "run_other" },
              ]),
            ),
          },
        },
      }),
    ],
  ])("rejects %s", (_, value) => {
    expect(() => requireAuditComposition(value, 200)).toThrow(
      expect.objectContaining({ code: "invalid_api_response" }),
    );
  });
});
