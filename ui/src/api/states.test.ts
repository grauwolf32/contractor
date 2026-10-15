import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import * as states from "./states";

const schemas = (
  parse(readFileSync("../api/openapi/contractor-public-v1.yaml", "utf8")) as {
    components: { schemas: Record<string, { enum?: string[] }> };
  }
).components.schemas;

function contractEnum(name: string): string[] {
  const values = schemas[name]?.enum;
  if (values === undefined) throw new Error(`OpenAPI has no enum ${name}`);
  return values;
}

const lists = Object.entries(states).flatMap(([name, value]) =>
  Array.isArray(value) ? [[name, value as readonly string[]] as const] : [],
);

describe("state lists", () => {
  it("read the full enums from the OpenAPI document", () => {
    expect(states.AUDIT_STATES).toEqual(contractEnum("AuditState"));
    expect(states.RUN_STATES).toEqual(contractEnum("WorkflowRunState"));
    expect(states.NON_TERMINAL_RUN_STATES).toEqual(
      contractEnum("NonTerminalWorkflowRunState"),
    );
    expect([
      ...states.NON_TERMINAL_RUN_STATES,
      ...states.TERMINAL_RUN_STATES,
    ]).toEqual(expect.arrayContaining(contractEnum("WorkflowRunState")));
    expect(states.TERMINAL_RUN_STATES).toEqual([
      "succeeded",
      "failed",
      "cancelled",
    ]);
  });

  it.each(lists)(
    "%s names distinct states of one OpenAPI enum",
    (name, list) => {
      const enumName = name.includes("RUN_STATES")
        ? "WorkflowRunState"
        : "AuditState";
      expect(list.length).toBeGreaterThan(0);
      expect(new Set(list).size).toBe(list.length);
      for (const state of list) {
        expect(contractEnum(enumName), `${name}: ${state}`).toContain(state);
      }
    },
  );

  it("keeps the Audit subsets consistent with each other", () => {
    expect(["draft", ...states.AUDIT_STATES_WITH_FINDINGS, "deleting"]).toEqual(
      expect.arrayContaining([...states.AUDIT_STATES]),
    );
    for (const subset of [
      states.WORKING_AUDIT_STATES,
      states.INBOX_AUDIT_STATES,
      states.OPEN_AUDIT_STATES,
      states.REPORT_AUDIT_STATES,
    ]) {
      expect(states.AUDIT_STATES_WITH_FINDINGS).toEqual(
        expect.arrayContaining([...subset]),
      );
    }
    expect(states.OPEN_AUDIT_STATES).toEqual(
      expect.arrayContaining([...states.INBOX_AUDIT_STATES]),
    );
    expect(states.ENDED_AUDIT_STATES).toEqual(
      expect.arrayContaining([...states.STOPPED_AUDIT_STATES]),
    );
  });
});
