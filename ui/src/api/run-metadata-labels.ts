const RUN_METADATA_LABEL_KEY = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
const encoder = new TextEncoder();

export const RUN_METADATA_LABEL_LIMIT = 32;
export const RUN_METADATA_LABEL_KEY_BYTE_LIMIT = 63;
export const RUN_METADATA_LABEL_VALUE_BYTE_LIMIT = 256;

export interface RunMetadataLabelDraft {
  id: string;
  key: string;
  value: string;
}

export interface RunMetadataLabelSelector {
  key: string;
  value: string;
}

export function runMetadataLabelKeyError(key: string): string | undefined {
  if (key.length === 0) {
    return "Label key is required.";
  }
  if (encoder.encode(key).length > RUN_METADATA_LABEL_KEY_BYTE_LIMIT) {
    return `Label key must be at most ${RUN_METADATA_LABEL_KEY_BYTE_LIMIT} UTF-8 bytes.`;
  }
  if (key.startsWith("contractor.")) {
    return "The contractor. prefix is reserved.";
  }
  if (!RUN_METADATA_LABEL_KEY.test(key)) {
    return "Use lowercase ASCII segments separated by ., _ or -.";
  }
  return undefined;
}

export function runMetadataLabelValueError(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return "Label value must be a string.";
  }
  const length = encoder.encode(value).length;
  if (value.includes("\0")) {
    return "Label value cannot contain U+0000.";
  }
  if (length > RUN_METADATA_LABEL_VALUE_BYTE_LIMIT) {
    return `Label value must be at most ${RUN_METADATA_LABEL_VALUE_BYTE_LIMIT} UTF-8 bytes.`;
  }
  return undefined;
}

function validateRunMetadataLabel(key: string, value: unknown): string {
  const keyError = runMetadataLabelKeyError(key);
  const valueError = runMetadataLabelValueError(value);
  if (keyError !== undefined || valueError !== undefined) {
    throw new TypeError(
      `Run metadata label ${JSON.stringify(key)} is invalid: ${keyError ?? valueError}`,
    );
  }
  return value as string;
}

export function safeRunMetadataLabels(value: unknown): Record<string, string> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Run metadata labels must be an object");
  }
  const entries = Object.entries(value);
  if (entries.length > RUN_METADATA_LABEL_LIMIT) {
    throw new TypeError(
      `Run metadata labels exceed ${RUN_METADATA_LABEL_LIMIT} entries`,
    );
  }
  entries.sort(([left], [right]) =>
    left === right ? 0 : left < right ? -1 : 1,
  );
  const result: Record<string, string> = {};
  for (const [key, child] of entries) {
    result[key] = validateRunMetadataLabel(key, child);
  }
  return result;
}

export function normalizeRunMetadataLabelSelectors(
  source: readonly RunMetadataLabelSelector[],
): RunMetadataLabelSelector[] {
  if (source.length > RUN_METADATA_LABEL_LIMIT) {
    throw new TypeError(
      `Run metadata label selectors exceed ${RUN_METADATA_LABEL_LIMIT} entries`,
    );
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
