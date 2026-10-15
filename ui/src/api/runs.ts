import { RESOURCE_ID } from "./contract";
import { isStateIn, TERMINAL_RUN_STATES } from "./states";
import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_PAGE_SIZE,
  ARTIFACT_REVISION_PATTERN,
  downloadExactArtifact,
  previewExactArtifact,
  requireRequestedMetadata,
  type ArtifactDetailRequest,
  type ArtifactMetadata,
  type ArtifactPage,
  type DownloadedArtifact,
} from "./artifacts";
import type { PublicAPI } from "./client";
import {
  invalidAPIResponse,
  PublicAPIError,
  publicAPIError,
  requireData,
} from "./error";
import type { components } from "./generated/public";
import { consistentAllocationResourceSummary } from "./performance";
import {
  canonicalOrder,
  canonicalRunRuntimeConfiguration,
  canonicalStageRuntimeConfiguration,
} from "./runtime-configuration";
import {
  normalizeRunMetadataLabelSelectors,
  type RunMetadataLabelSelector,
} from "./run-metadata-labels";

export const RUN_PAGE_SIZE = 50;
// Run and Stage execution IDs are public ResourceIds.
export const RUN_ID_PATTERN = RESOURCE_ID;
export { RUN_STATES, TERMINAL_RUN_STATES } from "./states";

export type WorkflowRunState = components["schemas"]["WorkflowRunState"];
export type WorkflowRunLifecycle =
  components["schemas"]["WorkflowRunLifecycle"];
export type RunPage = components["schemas"]["RunPage"];
export type RunStatus = components["schemas"]["RunStatus"];
export type RunSummary = components["schemas"]["RunSummary"];
export type StageAttempt = components["schemas"]["StageAttempt"];
export type StageTransition = components["schemas"]["StageTransition"];
export type PlannerPlan = components["schemas"]["PlannerPlan"];
export type CancelRunResponse = components["schemas"]["CancelRunResponse"];
export type ArtifactLineagePage = components["schemas"]["ArtifactLineagePage"];
export type RunRepeatDraftResponse =
  components["schemas"]["RunRepeatDraftResponse"];

export interface RunPageRequest {
  state?: WorkflowRunState;
  lifecycle?: WorkflowRunLifecycle;
  cursor?: string;
  labelSelectors?: readonly RunMetadataLabelSelector[];
}

export interface RunArtifactPageRequest {
  runId: string;
  namespace?: string;
  cursor?: string;
}

export interface RunArtifactDetailRequest extends ArtifactDetailRequest {
  runId: string;
}

export async function resumeRun(
  api: PublicAPI,
  runId: string,
  stageExecutionId: string,
): Promise<void> {
  requireRunID(runId);
  requireRunID(stageExecutionId);
  const result = await api.request((client) =>
    client.POST("/v1/runs/{runId}/resume", {
      params: { path: { runId } },
      body: { stageExecutionId },
    }),
  );
  const value = requireData(result);
  // The Server answers for the Run and Stage execution it was asked to resume.
  if (
    value.runId !== runId ||
    value.sourceStageExecutionId !== stageExecutionId
  ) {
    throw invalidRunResponse(result.response.status);
  }
}

function requireRunID(runId: string): void {
  if (!RUN_ID_PATTERN.test(runId)) {
    throw new TypeError("Run ID is invalid");
  }
}

function requireArtifactIdentity(namespace: string, name: string): void {
  if (
    !ARTIFACT_NAME_PATTERN.test(namespace) ||
    !ARTIFACT_NAME_PATTERN.test(name)
  ) {
    throw new TypeError("Run Artifact identity is invalid");
  }
}

function requireRevision(revision: string): void {
  if (!ARTIFACT_REVISION_PATTERN.test(revision)) {
    throw new TypeError("Run Artifact revision is invalid");
  }
}

function invalidRunResponse(status: number): PublicAPIError {
  return invalidAPIResponse(status, "Server returned an invalid Run response");
}

function runArtifactPath(
  runId: string,
  namespace: string,
  name: string,
): string {
  requireRunID(runId);
  requireArtifactIdentity(namespace, name);
  return `/v1/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
}

function exactRevisionQuery(revision: string): string {
  requireRevision(revision);
  return `?revision=${encodeURIComponent(revision)}`;
}

export function isTerminalRunState(state: WorkflowRunState): boolean {
  return isStateIn(TERMINAL_RUN_STATES, state);
}

export async function listRuns(
  api: PublicAPI,
  request: RunPageRequest = {},
): Promise<RunPage> {
  const labelSelectors = normalizeRunMetadataLabelSelectors(
    request.labelSelectors ?? [],
  );
  const result = await api.request((client) =>
    client.GET("/v1/runs", {
      params: {
        query: {
          limit: RUN_PAGE_SIZE,
          ...(request.state === undefined ? {} : { state: request.state }),
          ...(request.lifecycle === undefined
            ? {}
            : { lifecycle: request.lifecycle }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
          ...(labelSelectors.length === 0
            ? {}
            : {
                label: labelSelectors.map(
                  ({ key, value }) => `${key}=${value}`,
                ),
              }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function getRun(
  api: PublicAPI,
  runId: string,
): Promise<RunStatus> {
  requireRunID(runId);
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}", { params: { path: { runId } } }),
  );
  const run = requireData(result);
  if (
    // The Server answers for the requested Run.
    run.runId !== runId ||
    // Runtime labels and pins are in the Server's canonical order.
    !canonicalOrder(run.runtimeLabels) ||
    !canonicalRunRuntimeConfiguration(run.runtimeConfiguration) ||
    run.attempts.some(
      (attempt) =>
        (attempt.runtimeConfiguration !== undefined &&
          !canonicalStageRuntimeConfiguration(attempt.runtimeConfiguration)) ||
        // A resource summary describes the attempt that carries it.
        (attempt.resources ?? []).some(
          (resource) =>
            resource.runId !== run.runId ||
            resource.stageExecutionId !== attempt.stageExecutionId ||
            resource.stage !== attempt.stage ||
            resource.outcome !== attempt.state ||
            !consistentAllocationResourceSummary(resource),
        ),
    )
  ) {
    throw invalidRunResponse(result.response.status);
  }
  return run;
}

export async function getRunRepeatDraft(
  api: PublicAPI,
  runId: string,
): Promise<RunRepeatDraftResponse> {
  requireRunID(runId);
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}/repeat-draft", {
      params: { path: { runId } },
    }),
  );
  const repeat = requireData(result);
  const draft = repeat.draft;
  if (
    // The draft repeats the requested Run.
    repeat.sourceRunId !== runId ||
    // Audit-managed Runs never offer a draft; an ordinary Run without one
    // says why with a blocking notice.
    (repeat.authority === "audit-managed" && draft !== undefined) ||
    (repeat.authority === "ordinary" &&
      draft === undefined &&
      !repeat.notices.some((notice) => notice.severity === "blocking")) ||
    (draft !== undefined &&
      (!canonicalOrder(draft.runtimeLabels) ||
        // An available input names the exact revision its metadata describes.
        Object.values(draft.inputs).some(
          (input) =>
            input.status === "available" &&
            (input.artifact === undefined ||
              input.metadata === undefined ||
              input.sourceScope === undefined ||
              input.artifact.namespace !== input.metadata.artifact.namespace ||
              input.artifact.name !== input.metadata.artifact.name ||
              input.artifact.revision !== input.metadata.artifact.revision),
        )))
  ) {
    throw invalidRunResponse(result.response.status);
  }
  return repeat;
}

export async function cancelRun(
  api: PublicAPI,
  runId: string,
  reason: string,
): Promise<CancelRunResponse> {
  requireRunID(runId);
  const normalizedReason = reason.trim();
  if (normalizedReason.length === 0 || normalizedReason.length > 4096) {
    throw new TypeError("Cancellation reason must contain 1–4096 characters");
  }
  const result = await api.request((client) =>
    client.POST("/v1/runs/{runId}/cancel", {
      params: { path: { runId } },
      body: { reason: normalizedReason },
    }),
  );
  const response = requireData(result);
  // The Server answers for the Run it was asked to cancel.
  if (response.runId !== runId) {
    throw invalidRunResponse(result.response.status);
  }
  return response;
}

export async function deleteRun(api: PublicAPI, runId: string): Promise<void> {
  requireRunID(runId);
  const result = await api.request((client) =>
    client.DELETE("/v1/runs/{runId}", {
      params: { path: { runId } },
    }),
  );
  if (!result.response.ok) {
    throw publicAPIError(result.response.status, result.error);
  }
}

export async function listRunArtifacts(
  api: PublicAPI,
  request: RunArtifactPageRequest,
): Promise<ArtifactPage> {
  requireRunID(request.runId);
  if (
    request.namespace !== undefined &&
    !ARTIFACT_NAME_PATTERN.test(request.namespace)
  ) {
    throw new TypeError("Run Artifact namespace is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}/artifacts", {
      params: {
        path: { runId: request.runId },
        query: {
          limit: ARTIFACT_PAGE_SIZE,
          ...(request.namespace === undefined
            ? {}
            : { namespace: request.namespace }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function getRunArtifactMetadata(
  api: PublicAPI,
  request: RunArtifactDetailRequest,
): Promise<ArtifactMetadata> {
  requireRunID(request.runId);
  requireArtifactIdentity(request.namespace, request.name);
  if (request.revision !== undefined) {
    requireRevision(request.revision);
  }
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}/artifacts/{namespace}/{name}/metadata", {
      params: {
        path: {
          runId: request.runId,
          namespace: request.namespace,
          name: request.name,
        },
        query:
          request.revision === undefined ? {} : { revision: request.revision },
      },
    }),
  );
  return requireRequestedMetadata(
    requireData(result),
    request,
    result.response.status,
  );
}

export async function listRunArtifactVersions(
  api: PublicAPI,
  request: RunArtifactDetailRequest & { cursor?: string },
): Promise<ArtifactPage> {
  requireRunID(request.runId);
  requireArtifactIdentity(request.namespace, request.name);
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}/artifacts/{namespace}/{name}/versions", {
      params: {
        path: {
          runId: request.runId,
          namespace: request.namespace,
          name: request.name,
        },
        query: {
          limit: ARTIFACT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function getRunArtifactLineage(
  api: PublicAPI,
  request: RunArtifactDetailRequest & { cursor?: string },
): Promise<ArtifactLineagePage> {
  requireRunID(request.runId);
  requireArtifactIdentity(request.namespace, request.name);
  if (request.revision !== undefined) {
    requireRevision(request.revision);
  }
  const result = await api.request((client) =>
    client.GET("/v1/runs/{runId}/artifacts/{namespace}/{name}/lineage", {
      params: {
        path: {
          runId: request.runId,
          namespace: request.namespace,
          name: request.name,
        },
        query: {
          limit: ARTIFACT_PAGE_SIZE,
          ...(request.revision === undefined
            ? {}
            : { revision: request.revision }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function downloadRunArtifact(
  api: PublicAPI,
  runId: string,
  metadata: ArtifactMetadata,
): Promise<DownloadedArtifact> {
  const path = `${runArtifactPath(
    runId,
    metadata.artifact.namespace,
    metadata.artifact.name,
  )}${exactRevisionQuery(metadata.artifact.revision)}`;
  return downloadExactArtifact(api, metadata, path);
}

export async function previewRunArtifact(
  api: PublicAPI,
  runId: string,
  metadata: ArtifactMetadata,
): Promise<string> {
  return previewExactArtifact(metadata, () =>
    downloadRunArtifact(api, runId, metadata),
  );
}

export async function retryRunGateway(
  api: PublicAPI,
  runId: string,
): Promise<void> {
  requireRunID(runId);
  const result = await api.request((client) =>
    client.POST("/v1/runs/{runId}/retry-gateway", {
      params: { path: { runId } },
      body: {},
    }),
  );
  // The Server answers for the Run whose Gateway wait it retries.
  if (requireData(result).runId !== runId)
    throw invalidRunResponse(result.response.status);
}
