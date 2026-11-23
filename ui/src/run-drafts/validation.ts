import type { SharedConfig } from "../api/shared-config";
import type { ArtifactMetadata, ExactArtifactRef } from "../api/artifacts";
import { labelPlacementError } from "../api/audit-labels";
import {
  runMetadataLabelKeyError,
  runMetadataLabelValueError,
  type RunMetadataLabelDraft,
} from "../api/run-metadata-labels";
import {
  canonicalRuntimeLabels,
  RUNTIME_LABEL_LIMIT,
} from "../api/runtime-configuration";
import type { CreateRunRequest, WorkflowResource } from "../api/workflows";

interface RunDraftValues {
  runtimeLabels: string[];
  metadataLabels: readonly RunMetadataLabelDraft[];
  parameters: Record<string, string | undefined>;
  artifacts: Record<string, string>;
  artifactReviews: Readonly<Record<string, string>>;
}

interface RunDraftValidation {
  request?: CreateRunRequest;
  errors: Record<string, string>;
}

export function artifactOptionKey(ref: ExactArtifactRef): string {
  return `${ref.namespace}/${ref.name}@${ref.revision}`;
}

export function artifactAccepts(
  accepted: string[],
  metadata: ArtifactMetadata,
): boolean {
  return accepted.includes("*/*") || accepted.includes(metadata.mediaType);
}

export function validateRunDraft(
  workflow: WorkflowResource,
  values: RunDraftValues,
  artifactsForSlot: (
    slot: string,
  ) => ReadonlyMap<string, ArtifactMetadata> | undefined,
  limits: SharedConfig["limits"],
): RunDraftValidation {
  const errors: Record<string, string> = {};
  const runtimeLabels = canonicalRuntimeLabels(values.runtimeLabels);
  if (runtimeLabels === undefined) {
    errors.runtimeLabels = `Runtime labels must be a set of at most ${RUNTIME_LABEL_LIMIT} valid labels without default.`;
  }

  if (values.metadataLabels.length > limits.runMetadataMaxLabels) {
    errors.metadataLabels = `A Run can have at most ${limits.runMetadataMaxLabels} metadata labels.`;
  }
  const metadataLabels: Record<string, string> = {};
  const firstLabelByKey = new Map<string, string>();
  for (const label of values.metadataLabels) {
    const keyError = runMetadataLabelKeyError(label.key, limits);
    const valueError = runMetadataLabelValueError(label.value, limits);
    if (keyError !== undefined) {
      errors[`metadataLabel:${label.id}:key`] = keyError;
    }
    if (valueError !== undefined) {
      errors[`metadataLabel:${label.id}:value`] = valueError;
    }
    if (keyError !== undefined) {
      continue;
    }
    const firstID = firstLabelByKey.get(label.key);
    if (firstID !== undefined) {
      const duplicateError = `Label key ${label.key} is duplicated.`;
      errors[`metadataLabel:${firstID}:key`] = duplicateError;
      errors[`metadataLabel:${label.id}:key`] = duplicateError;
      continue;
    }
    firstLabelByKey.set(label.key, label.id);
    if (valueError === undefined) {
      metadataLabels[label.key] = label.value;
    }
  }
  const sortedMetadataLabels = Object.fromEntries(
    Object.entries(metadataLabels).sort(([left], [right]) =>
      left === right ? 0 : left < right ? -1 : 1,
    ),
  );
  const placementError = labelPlacementError(sortedMetadataLabels);
  if (placementError !== undefined) errors.metadataLabels = placementError;
  const parameters: Record<string, string> = {};
  for (const [name, slot] of Object.entries(workflow.parameters).sort()) {
    const value = values.parameters[name];
    if (value === undefined) {
      if (slot.required) {
        errors[`parameter:${name}`] = "Required string parameter is missing.";
      }
      continue;
    }
    parameters[name] = value;
  }

  const inputRefs: Record<string, ExactArtifactRef> = {};
  for (const [name, slot] of Object.entries(workflow.inputs).sort()) {
    const key = values.artifacts[name] ?? "";
    if (key === "") {
      if (slot.required) {
        errors[`artifact:${name}`] =
          "Choose a revision for this required input.";
      }
      continue;
    }
    const metadata = artifactsForSlot(name)?.get(key);
    if (metadata === undefined) {
      errors[`artifact:${name}`] =
        "The selected revision is no longer in the loaded list.";
      continue;
    }
    if (!artifactAccepts(slot.mediaTypes, metadata)) {
      errors[`artifact:${name}`] =
        `Media type ${metadata.mediaType} is not accepted by this input.`;
      continue;
    }
    if (values.artifactReviews[name] !== key) {
      errors[`artifactReview:${name}`] =
        "Review and confirm this exact revision for this input.";
      continue;
    }
    inputRefs[name] = metadata.artifact;
  }

  if (Object.keys(errors).length > 0 || runtimeLabels === undefined) {
    return { errors };
  }
  return {
    errors,
    request: {
      workflow: `${workflow.ref.name}@${workflow.ref.version}`,
      runtimeLabels,
      ...(Object.keys(sortedMetadataLabels).length === 0
        ? {}
        : { labels: sortedMetadataLabels }),
      parameters,
      artifacts: inputRefs,
    },
  };
}
