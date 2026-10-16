import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

import eventDocument from "../../../api/events/contractor-events-v1.schema.json";
import {
  conformsTo,
  eventFrameViolation,
  htmlPattern,
  responseViolation,
  schemaEnum,
  schemaLimit,
  stringRule,
} from "./contract";
import eventSchemas from "./generated/event-schemas.json";
import publicSchemas from "./generated/public-schemas.json";
import {
  RESPONSE_SCHEMAS,
  type PublicSchemaName,
} from "./generated/response-schemas";

const openapi = parse(
  readFileSync("../api/openapi/contractor-public-v1.yaml", "utf8"),
) as {
  paths: Record<string, Record<string, { responses?: object }>>;
  components: { schemas: Record<string, { example?: unknown }> };
};

const ANNOTATIONS = [
  "description",
  "title",
  "example",
  "examples",
  "x-enum-varnames",
  "$comment",
  "deprecated",
];

/** Visits every subschema at a schema-keyword position. */
function visitSchemas(
  schema: unknown,
  visit: (schema: Record<string, unknown>) => void,
): void {
  if (typeof schema !== "object" || schema === null) return;
  const value = schema as Record<string, unknown>;
  visit(value);
  for (const key of ["properties", "patternProperties", "$defs"]) {
    for (const child of Object.values((value[key] ?? {}) as object)) {
      visitSchemas(child, visit);
    }
  }
  for (const key of ["allOf", "anyOf", "oneOf", "prefixItems"]) {
    for (const child of (value[key] ?? []) as unknown[]) {
      visitSchemas(child, visit);
    }
  }
  for (const key of [
    "items",
    "additionalProperties",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
  ]) {
    visitSchemas(value[key], visit);
  }
}

describe("generated contract bundle", () => {
  it("holds every OpenAPI component schema", () => {
    for (const name of Object.keys(openapi.components.schemas)) {
      expect(publicSchemas.$defs, name).toHaveProperty([name]);
    }
  });

  it("strips annotations only at schema positions", () => {
    for (const bundle of [publicSchemas, eventSchemas]) {
      visitSchemas(bundle, (schema) => {
        for (const keyword of ANNOTATIONS) {
          expect(schema).not.toHaveProperty([keyword]);
        }
      });
    }
    const project = publicSchemas.$defs.Project as {
      properties: Record<string, unknown>;
    };
    expect(project.properties).toHaveProperty(["description"]);
    expect(stringRule("LoginRequest", "properties", "password")).toMatchObject({
      minBytes: 12,
      maxBytes: 1024,
    });
  });

  it("compiles every pattern with the Unicode flag", () => {
    for (const bundle of [publicSchemas, eventSchemas]) {
      visitSchemas(bundle, (schema) => {
        if (typeof schema.pattern === "string") {
          expect(() => new RegExp(schema.pattern as string, "u")).not.toThrow();
        }
      });
    }
  });

  it("accepts every OpenAPI component example", () => {
    const examples = Object.entries(openapi.components.schemas).filter(
      ([, schema]) => schema.example !== undefined,
    );
    expect(examples.length).toBeGreaterThan(0);
    for (const [name, schema] of examples) {
      expect(conformsTo(name as PublicSchemaName, schema.example), name).toBe(
        true,
      );
    }
  });

  it("accepts every event schema example", () => {
    for (const example of eventDocument.examples) {
      expect(eventFrameViolation(example)).toBeUndefined();
    }
  });

  it("maps every OpenAPI operation and declared status", () => {
    for (const [path, item] of Object.entries(openapi.paths)) {
      for (const [method, operation] of Object.entries(item)) {
        if (operation.responses === undefined) continue;
        const key = `${method.toUpperCase()} ${path}`;
        expect(Object.keys(RESPONSE_SCHEMAS), key).toContain(key);
        expect(
          Object.keys(RESPONSE_SCHEMAS[key as keyof typeof RESPONSE_SCHEMAS]),
        ).toEqual(Object.keys(operation.responses));
      }
    }
  });
});

describe("responseViolation", () => {
  const principal = {
    userId: "admin",
    username: "admin",
    capabilities: ["user"],
  };
  const session = {
    principal,
    csrfToken: "c".repeat(43),
    idleExpiresAt: "2026-10-15T12:00:00Z",
    absoluteExpiresAt: "2026-10-15T20:00:00Z",
  };

  it("accepts a conforming body", () => {
    expect(conformsTo("AuthSession", session)).toBe(true);
    expect(
      responseViolation("GET /v1/auth/session", 200, session),
    ).toBeUndefined();
  });

  it.each([
    ["a missing required member", { ...session, csrfToken: undefined }, "at #"],
    ["an unknown member", { ...session, token: "secret" }, "at #/token"],
    [
      "a wrong enum value",
      { ...session, principal: { ...principal, capabilities: ["root"] } },
      "at #/principal/capabilities/0",
    ],
    [
      "a pattern mismatch",
      { ...session, principal: { ...principal, username: "-admin" } },
      "at #/principal/username",
    ],
  ])("rejects %s and names its location", (_, body, location) => {
    const violation = responseViolation(
      "GET /v1/auth/session",
      200,
      JSON.parse(JSON.stringify(body)),
    );
    expect(violation).toBe(`the body does not conform ${location}`);
  });

  it("rejects an undeclared status, a missing body and an undeclared body", () => {
    expect(responseViolation("GET /v1/auth/session", 202, session)).toBe(
      "status 202 is not declared",
    );
    expect(responseViolation("GET /v1/auth/session", 200, undefined)).toBe(
      "the body does not conform at #",
    );
    expect(
      responseViolation("DELETE /v1/runs/{runId}", 204, undefined),
    ).toBeUndefined();
    expect(responseViolation("DELETE /v1/runs/{runId}", 204, {})).toBe(
      "an undeclared body was returned",
    );
  });

  // Error responses are declared through aliases (NotFound refers to
  // Error404, which declares the body), so the generator follows $ref chains.
  it.each([
    "GET /v1/auth/session",
    "GET /v1/projects/{projectId}",
    "DELETE /v1/runs/{runId}",
    "GET /v1/runs/{runId}/artifacts/{namespace}/{name}",
    "PUT /v1/operations/settings/scheduler",
  ] as const)("declares the error envelope for every error of %s", (key) => {
    const declared: Readonly<Record<string, string | null>> =
      RESPONSE_SCHEMAS[key];
    const errors = Object.entries(declared).filter(
      ([status]) => Number(status) >= 400,
    );
    expect(errors.length).toBeGreaterThan(0);
    for (const [status, schema] of errors) {
      expect(schema, `${key} ${status}`).toBe("Error");
      expect(
        responseViolation(key, Number(status), {
          code: "not_found",
          message: "Not found",
          retryable: false,
          requestId: "request_1",
        }),
      ).toBeUndefined();
    }
  });
});

describe("htmlPattern", () => {
  /** Every `pattern` keyword of the public bundle. */
  function patterns(value: unknown, found = new Set<string>()): Set<string> {
    if (Array.isArray(value)) {
      for (const item of value) patterns(item, found);
    } else if (typeof value === "object" && value !== null) {
      for (const [key, member] of Object.entries(value)) {
        if (key === "pattern" && typeof member === "string") found.add(member);
        else patterns(member, found);
      }
    }
    return found;
  }

  const samples = [
    "",
    "a",
    "run_1",
    "A9-b.c_d",
    "-lead",
    "_under",
    ".dot",
    "has space",
    "naïve",
    "a/b",
    "sha256:" + "a".repeat(64),
    "1.2.3",
    "owasp-top10@2",
    "a".repeat(300),
    "x&&y",
    "[brackets]",
  ];

  it("turns every contract pattern into an attribute browsers accept as the schema does", () => {
    const all = [...patterns(publicSchemas)];
    expect(all.length).toBeGreaterThan(20);
    for (const source of all) {
      const pattern = new RegExp(source, "u");
      const attribute = htmlPattern({ pattern });
      expect(attribute, source).toBeDefined();
      const browser = new RegExp(`^(?:${attribute})$`, "v");
      for (const sample of samples) {
        expect(browser.test(sample), `${source} on ${sample}`).toBe(
          pattern.test(sample),
        );
      }
    }
  });

  it("escapes a class edge dash for the v flag and drops the anchors", () => {
    const username = stringRule("LoginRequest", "properties", "username");
    expect(htmlPattern(username)).toBe("[A-Za-z0-9][A-Za-z0-9_.\\-]*");
    expect(htmlPattern({ pattern: /a[-x]/u })).toBe(".*a[\\-x].*");
    expect(htmlPattern({})).toBeUndefined();
  });
});

describe("contract rules", () => {
  it("reads string rules, enums and limits from the bundle", () => {
    const resourceId = stringRule("ResourceId");
    expect(resourceId.test("run_1")).toBe(true);
    expect(resourceId.test("a".repeat(256))).toBe(true);
    expect(resourceId.test("a".repeat(257))).toBe(false);
    expect(resourceId.test("_run")).toBe(false);
    expect(schemaEnum("ProjectKind")).toEqual(["project", "evaluation"]);
    expect(
      schemaLimit(
        "SchedulerSettings",
        "maximum",
        "properties",
        "maxConcurrentRuns",
      ),
    ).toBe(32);
    expect(() => stringRule("ProjectKind")).not.toThrow();
    expect(() => schemaEnum("ResourceId")).toThrow();
  });
});
