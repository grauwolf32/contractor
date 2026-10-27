import { describe, expect, it } from "vitest";
import fixture from "../../../../api/testdata/audit-check-plans/accepted-plan-cases.json";
import { at, patchDraft, renameBlock, starter } from "./document";
import { validateDraft } from "./validation";
import {
  CHECK_METHODS,
  MAX_CHECK_METHODS,
  MAX_CHECK_CAPABILITIES,
  FINDING_CHECK_PLAN_SCHEMA,
} from "./check-methods";

describe("registered check routes", () => {
  it("shares registered methods with Go and Runtime", () => {
    expect(MAX_CHECK_METHODS).toBe(fixture.maximumMethods);
    expect(MAX_CHECK_CAPABILITIES).toBe(fixture.maximumCapabilities);
    expect(CHECK_METHODS).toEqual(fixture.methods.map((entry) => entry.method));
    for (const method of fixture.methods) {
      let draft = patchDraft(
        starter("AuditProfile"),
        ["spec", "methods", "renamed-route"],
        {
          planSchema: FINDING_CHECK_PLAN_SCHEMA,
          method: method.method,
          workflowRole: "check",
          requiredCapabilities: [],
        },
      );
      expect(validateDraft(draft)).toEqual([]);
      draft = renameBlock(
        draft,
        ["spec", "workflows", "check"],
        "different-name",
      );
      expect(
        at(draft.value, ["spec", "methods", "renamed-route", "workflowRole"]),
      ).toBe("different-name");
      expect(validateDraft(draft)).toEqual([]);
      for (const phase of [method.phase, "guessed", null])
        expect(
          validateDraft(
            patchDraft(
              draft,
              ["spec", "methods", "renamed-route", "phase"],
              phase,
            ),
          ).length,
        ).toBeGreaterThan(0);
    }
  });
  it.each(fixture.authoredMappingCases)(
    "validates shared authored mapping $name without a phase field",
    ({ name, valid, mapping }) => {
      const draft = patchDraft(
        starter("AuditProfile"),
        ["spec", "methods", "route"],
        mapping,
      );
      expect(validateDraft(draft).length === 0, name).toBe(valid);
    },
  );
  it("enforces the shared capability tool limit through validation", () => {
    for (const count of [
      0,
      fixture.maximumCapabilityTools,
      fixture.maximumCapabilityTools + 1,
    ]) {
      const draft = patchDraft(
        starter("AuditProfile"),
        ["spec", "methods", "route"],
        {
          planSchema: FINDING_CHECK_PLAN_SCHEMA,
          method: "verify",
          workflowRole: "check",
          requiredCapabilities: [
            {
              ...fixture.base.plan.route.required_capabilities[0],
              tools: Array.from(
                { length: count },
                (_, index) => `tool${index}`,
              ),
            },
          ],
        },
      );
      expect(validateDraft(draft).length === 0, `${count} tools`).toBe(
        count === fixture.maximumCapabilityTools,
      );
    }
  });
  it("uses the shared blank-text and NUL rules for method names", () => {
    for (const test of fixture.mappingNameCases) {
      const draft = patchDraft(starter("AuditProfile"), ["spec", "methods"], {
        [test.value]: {
          planSchema: FINDING_CHECK_PLAN_SCHEMA,
          method: "verify",
          workflowRole: "check",
          requiredCapabilities: [],
        },
      });
      expect(validateDraft(draft).length === 0, test.name).toBe(test.valid);
      const capabilities = patchDraft(
        starter("AuditProfile"),
        ["spec", "methods", "route"],
        {
          planSchema: FINDING_CHECK_PLAN_SCHEMA,
          method: "verify",
          workflowRole: "check",
          requiredCapabilities: [
            {
              ...fixture.base.plan.route.required_capabilities[0],
              tools: [test.value],
            },
          ],
        },
      );
      expect(
        validateDraft(capabilities).length === 0,
        `tool ${test.name}`,
      ).toBe(test.valid);
    }
  });
  it("rejects unknown methods, duplicate semantic keys and non-check targets", () => {
    const mapping = {
      planSchema: FINDING_CHECK_PLAN_SCHEMA,
      method: "verify",
      workflowRole: "check",
      requiredCapabilities: [],
    };
    const base = patchDraft(
      starter("AuditProfile"),
      ["spec", "methods", "route"],
      mapping,
    );
    for (const [path, value] of [
      [["spec", "methods", "route", "method"], "free-text"],
      [["spec", "methods", "another"], mapping],
      [["spec", "methods", "route", "workflowRole"], "missing"],
      [["spec", "methods", "route", "requiredCapabilities"], null],
    ] as const)
      expect(
        validateDraft(patchDraft(base, [...path], value)).length,
      ).toBeGreaterThan(0);
  });
});
