const RUN_METADATA_LABEL_KEY = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
const encoder = new TextEncoder();

export function safeRunMetadataLabels(value: unknown): Record<string, string> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Run metadata labels must be an object");
  }
  const entries = Object.entries(value);
  if (entries.length > 32) {
    throw new TypeError("Run metadata labels exceed 32 entries");
  }
  entries.sort(([left], [right]) => left.localeCompare(right));
  const result: Record<string, string> = {};
  for (const [key, child] of entries) {
    if (
      encoder.encode(key).length > 63 ||
      !RUN_METADATA_LABEL_KEY.test(key) ||
      key.startsWith("contractor.") ||
      typeof child !== "string" ||
      encoder.encode(child).length === 0 ||
      encoder.encode(child).length > 256
    ) {
      throw new TypeError(
        `Run metadata label ${JSON.stringify(key)} is invalid`,
      );
    }
    result[key] = child;
  }
  return result;
}
