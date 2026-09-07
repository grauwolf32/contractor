import type { ArtifactMetadata } from "../api/artifacts";
import type { components } from "../api/generated/public";
import type { RunRepeatDraftResponse } from "../api/runs";
import type { RunMetadataLabelDraft } from "../api/run-metadata-labels";
import {
  initialRunDraftState,
  type RepeatDraftNotice,
  type RunDraftIdentity,
  type RunDraftState,
} from "./memory";
import {
  artifactOptionKey,
  NO_CREDENTIAL_OVERRIDE,
  type ConsumerOverrideDraft,
  type ExecutionOverrideDraft,
} from "./validation";

type SelectionPatch = components["schemas"]["ExecutionSelectionPatch"];

export interface PreparedRepeatDraft {
  identity: RunDraftIdentity;
  state: RunDraftState;
  destination: string;
}

function consumerDraft(patch?: SelectionPatch): ConsumerOverrideDraft {
  return {
    modelPolicy: patch?.modelPolicy ?? "",
    llmGateway: patch?.llmGateway ?? "",
    credential:
      patch?.credential === null
        ? NO_CREDENTIAL_OVERRIDE
        : (patch?.credential ?? ""),
  };
}

function metadataLabels(
  labels: Readonly<Record<string, string>>,
): RunMetadataLabelDraft[] {
  return Object.entries(labels)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value], index) => ({
      id: `repeat-${index + 1}`,
      key,
      value,
    }));
}

function destination(response: RunRepeatDraftResponse): string {
  const workflowPath = `${encodeURIComponent(response.workflow.name)}/${encodeURIComponent(response.workflow.version)}`;
  if (response.projectId === undefined) {
    return `/catalog/workflows/${workflowPath}#workflow-run-setup`;
  }
  return `/projects/${encodeURIComponent(response.projectId)}/workflows/${workflowPath}/run`;
}

export function auditDestination(
  response: RunRepeatDraftResponse,
): string | undefined {
  if (response.projectId === undefined || response.auditId === undefined) {
    return undefined;
  }
  return `/projects/${encodeURIComponent(response.projectId)}/audits/${encodeURIComponent(response.auditId)}`;
}

export function prepareRepeatDraft(
  response: RunRepeatDraftResponse,
): PreparedRepeatDraft | undefined {
  if (response.authority !== "ordinary" || response.draft === undefined) {
    return undefined;
  }
  const availableInputs = Object.entries(response.draft.inputs).filter(
    ([, input]) => input.status === "available" && input.metadata !== undefined,
  );
  const selections = Object.fromEntries(
    availableInputs.map(([slot, input]) => [
      slot,
      artifactOptionKey(input.metadata!.artifact),
    ]),
  );
  const knownArtifacts: ArtifactMetadata[] = availableInputs.map(([, input]) =>
    structuredClone(input.metadata!),
  );
  const retained = response.draft.executionConfig.value;
  const overrides: ExecutionOverrideDraft = {
    planner: consumerDraft(retained?.planner),
    workers: consumerDraft(retained?.workers),
    ...(retained?.stages === undefined
      ? {}
      : { stages: structuredClone(retained.stages) }),
  };
  const notices: RepeatDraftNotice[] = response.notices.map((notice) => ({
    ...notice,
  }));
  if (
    retained?.stages !== undefined &&
    Object.keys(retained.stages).length > 0
  ) {
    notices.push({
      code: "stage_execution_overrides_retained",
      severity: "warning",
      field: "executionConfig.stages",
      message:
        "Stage-specific execution overrides were retained exactly. They are shown read-only and can be removed as a group.",
    });
  }
  const state = initialRunDraftState(selections, knownArtifacts);
  state.parameters = { ...response.draft.parameters };
  state.runtimeLabels = [...response.draft.runtimeLabels];
  state.metadataLabels = metadataLabels(response.draft.labels);
  state.artifactReviews = {};
  state.overrides = overrides;
  state.repeat = {
    sourceRunId: response.sourceRunId,
    notices,
    reviewed: false,
  };
  return {
    identity: {
      workflowName: response.workflow.name,
      workflowVersion: response.workflow.version,
      ...(response.projectId === undefined
        ? {}
        : { projectId: response.projectId }),
    },
    state,
    destination: destination(response),
  };
}
