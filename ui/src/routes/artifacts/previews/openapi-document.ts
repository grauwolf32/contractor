import { parse } from "yaml";

const DANGEROUS_OBJECT_KEYS = new Set([
  "__proto__",
  "constructor",
  "prototype",
]);
const MARKDOWN_IMAGE_PATTERN =
  /!\[([^\]]*)\](?:\((?:\\.|[^)])*\)|\[[^\]]*\])?/g;
const HTML_EMBED_PATTERN =
  /<(?:audio|embed|iframe|img|object|source|video)\b[^>]*>/gi;
const MAXIMUM_DOCUMENT_DEPTH = 100;

function sanitiseText(value: string): string {
  return value
    .replace(MARKDOWN_IMAGE_PATTERN, (_match, alt: string) =>
      alt === "" ? "[image omitted]" : `[image omitted: ${alt}]`,
    )
    .replace(HTML_EMBED_PATTERN, "[embedded resource omitted]");
}

function sanitiseValue(
  value: unknown,
  ancestors: WeakSet<object>,
  depth: number,
): unknown {
  if (typeof value === "string") {
    return sanitiseText(value);
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (depth > MAXIMUM_DOCUMENT_DEPTH) {
    return "[nested value omitted]";
  }
  if (ancestors.has(value)) {
    return "[cyclic value omitted]";
  }

  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return value.map((item) => sanitiseValue(item, ancestors, depth + 1));
    }

    const result: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(value)) {
      const normalisedKey = key.toLowerCase();
      if (DANGEROUS_OBJECT_KEYS.has(key) || normalisedKey === "x-logo") {
        continue;
      }
      if (
        key === "$ref" &&
        typeof child === "string" &&
        !child.startsWith("#")
      ) {
        result.description = "External reference omitted from inline preview.";
        continue;
      }
      result[key] = sanitiseValue(child, ancestors, depth + 1);
    }
    return result;
  } finally {
    ancestors.delete(value);
  }
}

function isOpenApiDocument(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return (
    (typeof candidate.openapi === "string" &&
      /^3\.\d+\.\d+(?:[-+].*)?$/.test(candidate.openapi)) ||
    candidate.swagger === "2.0"
  );
}

export function parseOpenApiDocument(
  source: string,
): Record<string, unknown> | undefined {
  let parsed: unknown;
  try {
    parsed = parse(source, {
      maxAliasCount: 50,
      uniqueKeys: true,
    });
  } catch {
    return undefined;
  }
  if (!isOpenApiDocument(parsed)) {
    return undefined;
  }
  return sanitiseValue(parsed, new WeakSet(), 0) as Record<string, unknown>;
}
