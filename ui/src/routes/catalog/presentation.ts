/** Formatting only: exact names remain visible in the identity chip. */
export function libraryReadableName(name: string): string {
  return name
    .replaceAll(/[-_]+/g, " ")
    .replace(
      /\b(openapi|api|http|sqlmap|likec4)\b/gi,
      (word) =>
        ({
          openapi: "OpenAPI",
          api: "API",
          http: "HTTP",
          sqlmap: "SQLMap",
          likec4: "LikeC4",
        })[word.toLowerCase()] ?? word,
    )
    .replace(/^./, (char) => char.toUpperCase());
}
