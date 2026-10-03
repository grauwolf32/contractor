import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  MEDIA_TYPE_PATTERN,
} from "./artifacts";
import {
  AUDIT_STANDARD_SCHEME_PATTERN,
  AUDIT_STANDARD_VERSION_PATTERN,
} from "./audit-presets";
import { AUDIT_ID_PATTERN } from "./audits";
import { BUDGET_DURATION_PATTERN, CONFIGURATION_KINDS } from "./operations";
import { PROJECT_ID_PATTERN, PROJECT_REVISION_PATTERN } from "./projects";
import { QUEUE_MEMBERSHIPS, QUEUE_STATES } from "./queue";
import { RUN_ID_PATTERN, RUN_STATES, TERMINAL_RUN_STATES } from "./runs";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

interface StringSchema {
  pattern?: string;
  minLength?: number;
  maxLength?: number;
  enum?: string[];
}

const schemas = (
  parse(readFileSync("../api/openapi/contractor-public-v1.yaml", "utf8")) as {
    components: { schemas: Record<string, unknown> };
  }
).components.schemas;

// Resolve a dotted path such as "MediaType.oneOf.1" inside components.schemas.
function schemaAt(path: string): StringSchema {
  let value: unknown = schemas;
  for (const key of path.split(".")) {
    value = (value as Record<string, unknown>)[key];
  }
  if (typeof value !== "object" || value === null) {
    throw new Error(`OpenAPI has no schema at ${path}`);
  }
  return value as StringSchema;
}

function openapiAccepts(schema: StringSchema, value: string): boolean {
  const length = [...value].length;
  return (
    (schema.minLength === undefined || length >= schema.minLength) &&
    (schema.maxLength === undefined || length <= schema.maxLength) &&
    new RegExp(schema.pattern ?? "", "u").test(value)
  );
}

// Probe every printable ASCII character and a few non-ASCII and control
// characters in leading, inner and trailing positions, plus length bounds.
const probes = (() => {
  const characters = [
    ...Array.from({ length: 95 }, (_, index) =>
      String.fromCharCode(32 + index),
    ),
    "\n",
    "\t",
    "\u0000",
    " ",
    "é",
    "я",
  ];
  const values = new Set(["", "a", "A", "0", "1", "10", "01", "a/b", "a@b"]);
  for (const character of characters) {
    for (const value of [
      character,
      `a${character}`,
      `a${character}a`,
      `1${character}`,
      `1${character}1`,
      `a/${character}`,
      `a/a${character}`,
      `${character}/a`,
    ]) {
      values.add(value);
    }
  }
  for (const length of [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]) {
    values.add("a".repeat(length));
    values.add("1".repeat(length));
    values.add(`${"1".repeat(length)}s`);
    values.add(`${"1".repeat(length)}mo`);
    values.add(`a/${"b".repeat(length)}`);
  }
  return [...values];
})();

describe("hand-written UI contracts match the public OpenAPI", () => {
  it.each([
    ["ARTIFACT_NAME_PATTERN", ARTIFACT_NAME_PATTERN, "ArtifactName"],
    ["ARTIFACT_REVISION_PATTERN", ARTIFACT_REVISION_PATTERN, "Revision"],
    ["MEDIA_TYPE_PATTERN", MEDIA_TYPE_PATTERN, "MediaType.oneOf.1"],
    [
      "AUDIT_STANDARD_SCHEME_PATTERN",
      AUDIT_STANDARD_SCHEME_PATTERN,
      "AuditStandardRef.properties.scheme",
    ],
    [
      "AUDIT_STANDARD_VERSION_PATTERN",
      AUDIT_STANDARD_VERSION_PATTERN,
      "AuditStandardRef.properties.version",
    ],
    ["AUDIT_ID_PATTERN", AUDIT_ID_PATTERN, "ResourceId"],
    ["PROJECT_ID_PATTERN", PROJECT_ID_PATTERN, "ResourceId"],
    ["RUN_ID_PATTERN", RUN_ID_PATTERN, "ResourceId"],
    [
      "PROJECT_REVISION_PATTERN",
      PROJECT_REVISION_PATTERN,
      "Project.properties.revision",
    ],
    ["CONFIG_ID_PATTERN", CONFIG_ID_PATTERN, "ConfigId"],
    ["CONFIG_VERSION_PATTERN", CONFIG_VERSION_PATTERN, "ConfigVersion"],
    [
      "BUDGET_DURATION_PATTERN",
      BUDGET_DURATION_PATTERN,
      "GatewayPolicy.properties.budgetDuration",
    ],
  ])("%s accepts exactly what %s accepts", (_name, pattern, path) => {
    const schema = schemaAt(path as string);
    const mismatches = probes.filter(
      (value) =>
        (pattern as RegExp).test(value) !== openapiAccepts(schema, value),
    );
    expect(mismatches).toEqual([]);
  });

  it.each([
    ["RUN_STATES", RUN_STATES, "WorkflowRunState"],
    ["QUEUE_STATES", QUEUE_STATES, "NonTerminalWorkflowRunState"],
    ["QUEUE_MEMBERSHIPS", QUEUE_MEMBERSHIPS, "RunQueueMembership"],
    ["CONFIGURATION_KINDS", CONFIGURATION_KINDS, "ConfigurationKind"],
  ])("%s lists every %s value", (_name, values, path) => {
    expect([...(values as readonly string[])].sort()).toEqual(
      [...(schemaAt(path as string).enum ?? [])].sort(),
    );
  });

  it("TERMINAL_RUN_STATES are the run states the queue never shows", () => {
    const queued = new Set(schemaAt("NonTerminalWorkflowRunState").enum);
    expect([...TERMINAL_RUN_STATES].sort()).toEqual(
      (schemaAt("WorkflowRunState").enum ?? [])
        .filter((state) => !queued.has(state))
        .sort(),
    );
  });
});
