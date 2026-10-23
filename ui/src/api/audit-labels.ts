import { schemaLimit, schemaViolation } from "./contract";
import {
  runMetadataLabelKeyError,
  runMetadataLabelValueError,
} from "./run-metadata-labels";

export const AUDIT_LABEL_LIMIT = schemaLimit(
  "AuditLabels",
  "maxProperties",
  "allOf",
  "1",
);

/** Only these ordinary label keys constrain Runtime placement. */
export function labelPlacementError(
  labels: Readonly<Record<string, string>>,
): string | undefined {
  const required = ["cloud", "local"].flatMap((key) =>
    Object.hasOwn(labels, key)
      ? [labels[key] === "" ? key : `${key}/${labels[key]}`]
      : [],
  );
  if (schemaViolation("PlacementLabels", required) !== undefined)
    return "cloud and local values must be lowercase provider/site paths, such as aws or lab. Leave the value empty for the whole namespace.";
  if (required.length > 1)
    return "Choose cloud or local; both cannot apply to one launch.";
  return undefined;
}

/** Validate editable labels without dropping incomplete entries or values. */
export function auditLabels(
  entries: readonly { key: string; value: string }[],
): {
  labels: Record<string, string>;
  error: string | undefined;
} {
  const labels: Record<string, string> = {};
  for (const { key, value } of entries) {
    const error =
      runMetadataLabelKeyError(key) ?? runMetadataLabelValueError(value);
    if (error !== undefined) return { labels, error };
    if (key.startsWith("audit."))
      return {
        labels,
        error: "The audit. namespace is reserved for child Run provenance.",
      };
    if (Object.hasOwn(labels, key))
      return { labels, error: `Label ${key} is duplicated.` };
    labels[key] = value;
  }
  const maximum = AUDIT_LABEL_LIMIT;
  if (Object.keys(labels).length > maximum)
    return {
      labels,
      error: `Use at most ${maximum} labels; child Runs need room for audit provenance.`,
    };
  return { labels, error: labelPlacementError(labels) };
}
