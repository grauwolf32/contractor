import { parseDocument } from "yaml";

const MAXIMUM_FRONTMATTER_CHARACTERS = 16 * 1024;
const FIELDS = ["name", "description", "compatibility", "license"] as const;
type Field = (typeof FIELDS)[number];

export function splitFrontmatter(source: string): {
  body: string;
  fields: Partial<Record<Field, string>>;
  raw?: string;
  invalid?: true;
} {
  const start = /^(?:\uFEFF)?---[ \t]*\r?\n/.exec(source);
  if (start === null) return { body: source, fields: {} };
  const end = /^---[ \t]*\r?$/m.exec(
    source.slice(start[0].length, MAXIMUM_FRONTMATTER_CHARACTERS),
  );
  if (end === null) return { body: source, fields: {} };
  const raw = source.slice(start[0].length, start[0].length + end.index);
  if (!/^(?:name|description|compatibility|license):/m.test(raw))
    return { body: source, fields: {} };
  const body = source
    .slice(start[0].length + end.index + end[0].length)
    .replace(/^\r?\n/, "");
  try {
    const document = parseDocument(raw, {
      uniqueKeys: true,
      schema: "failsafe",
    });
    if (document.errors.length || document.warnings.length) throw new Error();
    const value: unknown = document.toJS({ maxAliasCount: 0 });
    if (value === null || typeof value !== "object" || Array.isArray(value))
      throw new Error();
    const fields: Partial<Record<Field, string>> = {};
    for (const field of FIELDS) {
      const candidate = Object.getOwnPropertyDescriptor(value, field)?.value;
      if (typeof candidate === "string" && candidate.trim() !== "")
        fields[field] = candidate
          .trim()
          .slice(0, field === "description" ? 4096 : 1024);
    }
    return { body, fields, raw };
  } catch {
    return { body, fields: {}, raw, invalid: true };
  }
}
