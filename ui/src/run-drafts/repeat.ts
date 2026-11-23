import type { RunRepeatDraftResponse } from "../api/runs";
import type { RunMetadataLabelDraft } from "../api/run-metadata-labels";
import {
  initialRunDraftState,
  type RepeatDraftNotice,
  type RunDraftIdentity,
  type RunDraftState,
} from "./memory";
import { artifactOptionKey } from "./validation";

interface PreparedRepeatDraft {
  identity: RunDraftIdentity;
  state: RunDraftState;
  destination: string;
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
  const knownArtifacts = Object.fromEntries(
    availableInputs.map(([slot, input]) => [
      slot,
      [structuredClone(input.metadata!)],
    ]),
  );
  const notices: RepeatDraftNotice[] = response.notices.map((notice) => ({
    ...notice,
  }));
  const state = initialRunDraftState(selections, knownArtifacts);
  state.parameters = { ...response.draft.parameters };
  state.runtimeLabels = [...response.draft.runtimeLabels];
  state.metadataLabels = metadataLabels(response.draft.labels);
  state.artifactReviews = {};
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
