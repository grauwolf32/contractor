const RUN_METADATA_LABEL_KEY = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
const encoder = new TextEncoder();

export interface RunMetadataLabelSelector {
  key: string;
  value: string;
}

function validateRunMetadataLabel(key: string, value: unknown): string {
  if (
    encoder.encode(key).length > 63 ||
    !RUN_METADATA_LABEL_KEY.test(key) ||
    key.startsWith("contractor.") ||
    typeof value !== "string" ||
    value.includes("\0") ||
    encoder.encode(value).length === 0 ||
    encoder.encode(value).length > 256
  ) {
    throw new TypeError(`Run metadata label ${JSON.stringify(key)} is invalid`);
  }
  return value;
}

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
    result[key] = validateRunMetadataLabel(key, child);
  }
  return result;
}

export function normalizeRunMetadataLabelSelectors(
  source: readonly RunMetadataLabelSelector[],
): RunMetadataLabelSelector[] {
  if (source.length > 32) {
    throw new TypeError("Run metadata label selectors exceed 32 entries");
  }
  const unique = new Map<string, RunMetadataLabelSelector>();
  for (const selector of source) {
    const normalized = {
      key: selector.key,
      value: validateRunMetadataLabel(selector.key, selector.value),
    };
    unique.set(JSON.stringify([normalized.key, normalized.value]), normalized);
  }
  return [...unique.values()].sort((left, right) => {
    if (left.key !== right.key) {
      return left.key < right.key ? -1 : 1;
    }
    return left.value === right.value ? 0 : left.value < right.value ? -1 : 1;
  });
}
