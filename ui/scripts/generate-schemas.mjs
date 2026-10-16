// Writes the response validation contract into src/api/generated (or the
// directory given as the first argument):
//
// - public-schemas.json: every OpenAPI component schema under $defs of one
//   bundle, plus the inline JSON response schemas, with annotations removed
//   and $ref values made absolute;
// - event-schemas.json: the contractor.events.v1 frame schema, likewise;
// - response-schemas.ts: for each operation ("METHOD /path/{template}") and
//   declared status, the $defs name of its JSON body schema, or null when the
//   status has no JSON body; and for each operation with header parameters,
//   the $defs name of each header's schema (added to the bundle as
//   <Parameter>Header), so the client checks If-Match and Idempotency-Key
//   values before it sends them;
// - *.d.json.ts: declarations for the two bundles, so that TypeScript does
//   not infer a literal type for 250 kB of JSON.
//
// @cfworker/json-schema interprets the bundles at run time (src/api/
// contract.ts). It ignores keywords it does not know, so the x-min-utf8-bytes
// and x-max-utf8-bytes extensions stay for request-side form checks.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { format } from "prettier";
import { parse } from "yaml";

export const PUBLIC_SCHEMA_ID =
  "https://contractor.local/schemas/contractor-public-v1.json";
export const EVENT_SCHEMA_ID =
  "https://contractor.local/schemas/contractor-events-v1.schema.json";

const OPENAPI = "../api/openapi/contractor-public-v1.yaml";
const EVENTS = "../api/events/contractor-events-v1.schema.json";
const METHODS = ["get", "put", "post", "delete", "patch"];
const ANNOTATIONS = new Set([
  "description",
  "title",
  "example",
  "examples",
  "x-enum-varnames",
  "$comment",
  "deprecated",
]);
// Keywords whose value is one schema, an array of schemas, or a map from
// names to schemas. Every other keyword holds data (enum, const, default,
// required, ...) and is copied without descending into it, so a property
// or $defs entry named like an annotation survives.
const SCHEMA = new Set([
  "additionalItems",
  "additionalProperties",
  "contains",
  "else",
  "if",
  "items",
  "not",
  "propertyNames",
  "then",
  "unevaluatedItems",
  "unevaluatedProperties",
]);
const SCHEMA_ARRAY = new Set(["allOf", "anyOf", "oneOf", "prefixItems"]);
const SCHEMA_MAP = new Set([
  "$defs",
  "definitions",
  "dependentSchemas",
  "patternProperties",
  "properties",
]);

function fail(message) {
  throw new Error(`generate-schemas: ${message}`);
}

/** Copies a schema without annotations, rewriting each $ref with `ref`. */
function clean(schema, ref, where) {
  if (typeof schema === "boolean") return schema;
  if (typeof schema !== "object" || schema === null || Array.isArray(schema)) {
    fail(`${where} is not a schema`);
  }
  const result = {};
  for (const [key, value] of Object.entries(schema)) {
    if (ANNOTATIONS.has(key)) continue;
    const at = `${where}/${key}`;
    if (key === "$ref") {
      result[key] = ref(value, at);
    } else if (SCHEMA_MAP.has(key)) {
      result[key] = Object.fromEntries(
        Object.entries(value).map(([name, child]) => [
          name,
          clean(child, ref, `${at}/${name}`),
        ]),
      );
    } else if (
      SCHEMA_ARRAY.has(key) ||
      (key === "items" && Array.isArray(value))
    ) {
      result[key] = value.map((child, index) =>
        clean(child, ref, `${at}/${index}`),
      );
    } else if (SCHEMA.has(key)) {
      result[key] = clean(value, ref, at);
    } else {
      result[key] = value;
    }
  }
  return result;
}

function publicRef(value, where) {
  const match = /^#\/components\/schemas\/([^/]+)$/.exec(value);
  if (match === null) fail(`${where} refers outside components.schemas`);
  return `${PUBLIC_SCHEMA_ID}#/$defs/${match[1]}`;
}

function eventRef(value, where) {
  const match = /^#\/\$defs\/([^/]+)$/.exec(value);
  if (match === null) fail(`${where} refers outside $defs`);
  return `${EVENT_SCHEMA_ID}#/$defs/${match[1]}`;
}

/**
 * The response a declared response stands for, following $ref chains
 * (NotFound refers to Error404, which declares the body).
 */
function resolveResponse(document, response, where) {
  const seen = new Set();
  let current = response;
  while (current.$ref !== undefined) {
    if (seen.has(current.$ref)) fail(`${where} has a cyclic response $ref`);
    seen.add(current.$ref);
    const match = /^#\/components\/responses\/([^/]+)$/.exec(current.$ref);
    const target = match && document.components.responses[match[1]];
    if (!target) fail(`${where} has an unresolved response $ref`);
    current = target;
  }
  return current;
}

/** The $defs name of a header parameter's schema, added on first use. */
function headerSchema(document, defs, operation, key, declared) {
  let parameter = declared;
  let name;
  if (declared.$ref !== undefined) {
    const match = /^#\/components\/parameters\/([^/]+)$/.exec(declared.$ref);
    parameter = match && document.components.parameters[match[1]];
    if (!parameter) fail(`${key} has an unresolved parameter $ref`);
    name = `${match[1]}Header`;
  } else {
    name = `${operation.operationId}${parameter.name.replace(/[^A-Za-z0-9]/g, "")}Header`;
  }
  if (parameter.in !== "header") return undefined;
  if (parameter.schema === undefined)
    fail(`${key} ${parameter.name} has no schema`);
  if (!(name in defs)) {
    if (name in document.components.schemas) {
      fail(`${key} ${parameter.name} collides with schema ${name}`);
    }
    defs[name] = clean(parameter.schema, publicRef, `${key} ${parameter.name}`);
  }
  return [parameter.name, name];
}

function publicBundle(document) {
  const defs = {};
  for (const [name, schema] of Object.entries(document.components.schemas)) {
    defs[name] = clean(schema, publicRef, `#/components/schemas/${name}`);
  }
  const operations = {};
  const headers = {};
  for (const [path, item] of Object.entries(document.paths)) {
    for (const method of METHODS) {
      const operation = item[method];
      if (operation === undefined) continue;
      const key = `${method.toUpperCase()} ${path}`;
      const statuses = {};
      for (const [status, declared] of Object.entries(operation.responses)) {
        if (!/^[1-5][0-9][0-9]$/.test(status)) {
          fail(`${key} declares unsupported status ${status}`);
        }
        const response = resolveResponse(
          document,
          declared,
          `${key} ${status}`,
        );
        const schema = response.content?.["application/json"]?.schema;
        if (schema === undefined) {
          statuses[status] = null;
        } else if (schema.$ref !== undefined) {
          statuses[status] = publicRef(schema.$ref, `${key} ${status}`).slice(
            PUBLIC_SCHEMA_ID.length + "#/$defs/".length,
          );
        } else {
          const name = `${operation.operationId}Response${status}`;
          if (operation.operationId === undefined || name in defs) {
            fail(`${key} ${status} needs a unique inline schema name`);
          }
          defs[name] = clean(schema, publicRef, `${key} ${status}`);
          statuses[status] = name;
        }
      }
      operations[key] = statuses;
      const declared = [
        ...(item.parameters ?? []),
        ...(operation.parameters ?? []),
      ];
      const operationHeaders = Object.fromEntries(
        declared
          .map((parameter) =>
            headerSchema(document, defs, operation, key, parameter),
          )
          .filter((entry) => entry !== undefined),
      );
      if (Object.keys(operationHeaders).length > 0) {
        headers[key] = operationHeaders;
      }
    }
  }
  return {
    bundle: {
      $schema: "https://json-schema.org/draft/2020-12/schema",
      $id: PUBLIC_SCHEMA_ID,
      $defs: defs,
    },
    operations,
    headers,
  };
}

function eventBundle(schema) {
  if (schema.$id !== EVENT_SCHEMA_ID) fail(`${EVENTS} changed its $id`);
  return clean(schema, eventRef, "#");
}

const DECLARATION = `// Generated by scripts/generate-schemas.mjs. Do not edit.
declare const bundle: {
  readonly $id: string;
  readonly $defs: Readonly<Record<string, unknown>>;
};
export default bundle;
`;

function responseModule(operations, headers, names, eventNames) {
  return `// Generated by scripts/generate-schemas.mjs from
// api/openapi/contractor-public-v1.yaml. Do not edit.

export const PUBLIC_SCHEMA_ID = ${JSON.stringify(PUBLIC_SCHEMA_ID)};
export const EVENT_SCHEMA_ID = ${JSON.stringify(EVENT_SCHEMA_ID)};

/** A schema under $defs of public-schemas.json. */
export type PublicSchemaName = ${names.map((name) => JSON.stringify(name)).join(" | ")};

/** A schema under $defs of event-schemas.json. */
export type EventSchemaName = ${eventNames.map((name) => JSON.stringify(name)).join(" | ")};

/**
 * Each operation's declared statuses and the PublicSchemaName of the JSON
 * body, or null for a status without a JSON body.
 */
export const RESPONSE_SCHEMAS = ${JSON.stringify(operations)} as const satisfies Record<string, Record<string, PublicSchemaName | null>>;

export type ResponseOperation = keyof typeof RESPONSE_SCHEMAS;

/** The PublicSchemaName of each request header an operation declares. */
export const REQUEST_HEADER_SCHEMAS = ${JSON.stringify(headers)} as const satisfies Partial<Record<ResponseOperation, Record<string, PublicSchemaName>>>;
`;
}

export async function generateSchemas(directory) {
  const document = parse(readFileSync(OPENAPI, "utf8"));
  const events = JSON.parse(readFileSync(EVENTS, "utf8"));
  const { bundle, operations, headers } = publicBundle(document);
  mkdirSync(directory, { recursive: true });
  const outputs = {
    "public-schemas.json": await format(JSON.stringify(bundle), {
      parser: "json",
    }),
    "event-schemas.json": await format(JSON.stringify(eventBundle(events)), {
      parser: "json",
    }),
    "public-schemas.d.json.ts": DECLARATION,
    "event-schemas.d.json.ts": DECLARATION,
    "response-schemas.ts": await format(
      responseModule(
        operations,
        headers,
        Object.keys(bundle.$defs),
        Object.keys(events.$defs),
      ),
      { parser: "typescript", trailingComma: "all" },
    ),
  };
  for (const [name, content] of Object.entries(outputs)) {
    writeFileSync(join(directory, name), content);
  }
  return Object.keys(outputs);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  await generateSchemas(process.argv[2] ?? "src/api/generated");
}
