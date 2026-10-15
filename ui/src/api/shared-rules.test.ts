import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import configIdentityCases from "../../../api/testdata/v1alpha1/config-identity-cases.json";
import digestCases from "../../../api/testdata/v1alpha1/digest-cases.json";
import capabilityCases from "../../../api/testdata/v1alpha1/runtime-capability-cases.json";
import {
  CONFIG_ID,
  CONFIG_VERSION,
  CONFIGURATION_NAME,
  DIGEST,
  RESOURCE_ID,
  REVISION,
  RUNTIME_AGENT_ID,
  RUNTIME_CONFIG_VERSION,
  RUNTIME_CREDENTIAL_ID,
  RUNTIME_INFRASTRUCTURE_ID,
  type StringRule,
} from "./contract";

interface StringSchema {
  pattern?: string;
  minLength?: number;
  maxLength?: number;
}

const schemas = (
  parse(readFileSync("../api/openapi/contractor-public-v1.yaml", "utf8")) as {
    components: { schemas: Record<string, StringSchema> };
  }
).components.schemas;

/** The raw OpenAPI schema's verdict, compiled here independently. */
function openapiAccepts(name: string, value: string): boolean {
  const schema = schemas[name];
  if (schema === undefined) throw new Error(`OpenAPI has no schema ${name}`);
  const length = [...value].length;
  return (
    (schema.minLength === undefined || length >= schema.minLength) &&
    (schema.maxLength === undefined || length <= schema.maxLength) &&
    new RegExp(schema.pattern ?? "", "u").test(value)
  );
}

function expectCases(
  rule: StringRule,
  schema: string,
  cases: { maxLength?: number; valid: string[]; invalid: string[] },
) {
  const valid = [...cases.valid];
  const invalid = [...cases.invalid];
  if (cases.maxLength !== undefined) {
    valid.push("a".repeat(cases.maxLength));
    invalid.push("a".repeat(cases.maxLength + 1));
  }
  for (const value of valid) {
    expect(rule.test(value), `valid ${JSON.stringify(value)}`).toBe(true);
    expect(openapiAccepts(schema, value), value).toBe(true);
  }
  for (const value of invalid) {
    expect(rule.test(value), `invalid ${JSON.stringify(value)}`).toBe(false);
    expect(openapiAccepts(schema, value), value).toBe(false);
  }
}

describe("shared rules from the generated contract", () => {
  it("ResourceId and Revision accept exactly the shared identity cases", () => {
    expectCases(RESOURCE_ID, "ResourceId", capabilityCases.identity);
    expectCases(REVISION, "Revision", capabilityCases.identity);
  });

  it("ConfigId accepts exactly the shared tool and Stage name cases", () => {
    expectCases(CONFIG_ID, "ConfigId", capabilityCases.toolName);
  });

  it("configuration names and versions accept exactly the shared cases", () => {
    expectCases(
      CONFIGURATION_NAME,
      "ConfigurationName",
      configIdentityCases.id,
    );
    expectCases(CONFIG_VERSION, "ConfigVersion", configIdentityCases.version);
  });

  it("Digest accepts every shared configuration digest", () => {
    const digests = Object.values(digestCases).flatMap((group) =>
      (group as { ref: { digest: string } }[]).map((entry) => entry.ref.digest),
    );
    expect(digests.length).toBeGreaterThan(0);
    for (const digest of digests) {
      expect(DIGEST.test(digest), digest).toBe(true);
    }
    for (const digest of [
      `sha256:${"A".repeat(64)}`,
      `sha256:${"a".repeat(63)}`,
      `sha512:${"a".repeat(64)}`,
      "a".repeat(64),
    ]) {
      expect(DIGEST.test(digest), digest).toBe(false);
    }
  });

  it("Runtime identities follow their schemas", () => {
    for (const [rule, schema] of [
      [RUNTIME_INFRASTRUCTURE_ID, "RuntimeInfrastructureId"],
      [RUNTIME_CREDENTIAL_ID, "RuntimeCredentialId"],
      [RUNTIME_CONFIG_VERSION, "RuntimeConfigVersion"],
    ] as const) {
      for (const value of [
        "a",
        "debug",
        "caido-proxy",
        "a_1",
        "1a",
        "A",
        "a.b",
        "a+b",
        "",
        "a".repeat(63),
        "a".repeat(64),
        "a".repeat(128),
        "a".repeat(129),
      ]) {
        expect(rule.test(value), `${schema} ${value}`).toBe(
          openapiAccepts(schema, value),
        );
      }
    }
    expect(RUNTIME_AGENT_ID.test("0".repeat(64))).toBe(true);
    expect(RUNTIME_AGENT_ID.test("0".repeat(63))).toBe(false);
    expect(RUNTIME_AGENT_ID.test("A".repeat(64))).toBe(false);
  });
});
