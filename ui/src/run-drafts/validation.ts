import type { ArtifactMetadata, ExactArtifactRef } from "../api/artifacts";
import type { components } from "../api/generated/public";
import type { CreateRunRequest, WorkflowResource } from "../api/workflows";

export const NO_CREDENTIAL_OVERRIDE = "__none__";

export interface ConsumerOverrideDraft {
  modelPolicy: string;
  llmGateway: string;
  credential: string;
}

export interface ExecutionOverrideDraft {
  planner: ConsumerOverrideDraft;
  workers: ConsumerOverrideDraft;
}

export interface RunDraftValues {
  runtimeLabels: string[];
  parameters: Record<string, string | undefined>;
  artifacts: Record<string, string>;
  overrides: ExecutionOverrideDraft;
}

export interface RunDraftValidation {
  request?: CreateRunRequest;
  errors: Record<string, string>;
}

type ExecutionSelectionPatch = components["schemas"]["ExecutionSelectionPatch"];

export function artifactOptionKey(ref: ExactArtifactRef): string {
  return `${ref.namespace}/${ref.name}@${ref.revision}`;
}

export function artifactAccepts(
  accepted: string[],
  metadata: ArtifactMetadata,
): boolean {
  return accepted.includes("*/*") || accepted.includes(metadata.mediaType);
}

function selectionPatch(
  draft: ConsumerOverrideDraft,
): ExecutionSelectionPatch | undefined {
  const result: ExecutionSelectionPatch = {};
  if (draft.modelPolicy !== "") {
    result.modelPolicy = draft.modelPolicy;
  }
  if (draft.llmGateway !== "") {
    result.llmGateway = draft.llmGateway;
  }
  if (draft.credential === NO_CREDENTIAL_OVERRIDE) {
    result.credential = null;
  } else if (draft.credential !== "") {
    result.credential = draft.credential;
  }
  return Object.keys(result).length === 0 ? undefined : result;
}

export function emptyExecutionOverrides(): ExecutionOverrideDraft {
  return {
    planner: { modelPolicy: "", llmGateway: "", credential: "" },
    workers: { modelPolicy: "", llmGateway: "", credential: "" },
  };
}

export function validateRunDraft(
  workflow: WorkflowResource,
  values: RunDraftValues,
  artifacts: ReadonlyMap<string, ArtifactMetadata>,
): RunDraftValidation {
  const errors: Record<string, string> = {};
  const runtimeLabels = [...values.runtimeLabels].sort();
  if (
    runtimeLabels.length > 32 ||
    runtimeLabels.includes("default") ||
    new Set(runtimeLabels).size !== runtimeLabels.length
  ) {
    errors.runtimeLabels =
      "Runtime labels must be a unique set without default.";
  }
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
        errors[`artifact:${name}`] = "Required Artifact input is missing.";
      }
      continue;
    }
    const metadata = artifacts.get(key);
    if (metadata === undefined) {
      errors[`artifact:${name}`] =
        "Selected Artifact revision is no longer in the loaded inventory.";
      continue;
    }
    if (!artifactAccepts(slot.mediaTypes, metadata)) {
      errors[`artifact:${name}`] =
        `Artifact media type ${metadata.mediaType} is not accepted.`;
      continue;
    }
    inputRefs[name] = metadata.artifact;
  }

  const planner = selectionPatch(values.overrides.planner);
  const workers = selectionPatch(values.overrides.workers);
  const executionConfig = {
    ...(planner === undefined ? {} : { planner }),
    ...(workers === undefined ? {} : { workers }),
  };
  if (Object.keys(errors).length > 0) {
    return { errors };
  }
  return {
    errors,
    request: {
      workflow: `${workflow.ref.name}@${workflow.ref.version}`,
      runtimeLabels,
      parameters,
      artifacts: inputRefs,
      ...(Object.keys(executionConfig).length === 0 ? {} : { executionConfig }),
    },
  };
}
