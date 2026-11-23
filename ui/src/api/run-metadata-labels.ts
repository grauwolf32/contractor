import type { SharedConfig } from "./shared-config";

const RUN_METADATA_LABEL_KEY = /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/;
const encoder = new TextEncoder();

type RunMetadataLimits = Pick<
  SharedConfig["limits"],
  "runMetadataMaxLabels" | "runMetadataKeyMaxBytes" | "runMetadataValueMaxBytes"
>;

export interface RunMetadataLabelDraft {
  id: string;
  key: string;
  value: string;
}

export interface RunMetadataLabelSelector {
  key: string;
  value: string;
}

export function runMetadataLabelKeyError(
  key: string,
  limits: RunMetadataLimits,
): string | undefined {
  if (key.length === 0) {
    return "Label key is required.";
  }
  if (encoder.encode(key).length > limits.runMetadataKeyMaxBytes) {
    return `Label key must be at most ${limits.runMetadataKeyMaxBytes} UTF-8 bytes.`;
  }
  if (key.startsWith("contractor.")) {
    return "The contractor. prefix is reserved.";
  }
  if (!RUN_METADATA_LABEL_KEY.test(key)) {
    return "Use lowercase ASCII segments separated by ., _ or -.";
  }
  return undefined;
}

export function runMetadataLabelValueError(
  value: unknown,
  limits: RunMetadataLimits,
): string | undefined {
  if (typeof value !== "string") {
    return "Label value must be a string.";
  }
  const length = encoder.encode(value).length;
  if (value.includes("\0")) {
    return "Label value cannot contain U+0000.";
  }
  if (length > limits.runMetadataValueMaxBytes) {
    return `Label value must be at most ${limits.runMetadataValueMaxBytes} UTF-8 bytes.`;
  }
  return undefined;
}

function validateRunMetadataLabel(
  key: string,
  value: unknown,
  limits: RunMetadataLimits,
): string {
  const keyError = runMetadataLabelKeyError(key, limits);
  const valueError = runMetadataLabelValueError(value, limits);
  if (keyError !== undefined || valueError !== undefined) {
    throw new TypeError(
      `Run metadata label ${JSON.stringify(key)} is invalid: ${keyError ?? valueError}`,
    );
  }
  return value as string;
}

export function normalizeRunMetadataLabelSelectors(
  source: readonly RunMetadataLabelSelector[],
  limits: RunMetadataLimits,
): RunMetadataLabelSelector[] {
  if (source.length > limits.runMetadataMaxLabels) {
    throw new TypeError(
      `Run metadata label selectors exceed ${limits.runMetadataMaxLabels} entries`,
    );
  }
  const unique = new Map<string, RunMetadataLabelSelector>();
  for (const selector of source) {
    const normalized = {
      key: selector.key,
      value: validateRunMetadataLabel(selector.key, selector.value, limits),
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
