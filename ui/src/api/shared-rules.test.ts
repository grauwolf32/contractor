import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import capabilityCases from "../../../api/testdata/v1alpha1/runtime-capability-cases.json";
import { RESOURCE_ID, REVISION, type StringRule } from "./contract";

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
    // runtime-capability-cases.json "identity" is the public ResourceId
    // grammar; Go and the Runtime test the same list.
    expectCases(RESOURCE_ID, "ResourceId", capabilityCases.identity);
    expectCases(REVISION, "Revision", capabilityCases.identity);
  });
});
