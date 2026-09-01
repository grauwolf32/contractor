import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_PAGE_SIZE,
  ARTIFACT_REVISION_PATTERN,
  downloadExactArtifact,
  previewExactArtifact,
  type ArtifactDetailRequest,
  type ArtifactMetadata,
  type ArtifactPage,
  type DownloadedArtifact,
} from "./artifacts";
import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import {
  safeRunRuntimeConfiguration,
  safeStageRuntimeConfiguration,
} from "./runtime-configuration";

export const RUN_PAGE_SIZE = 50;
export const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
export const RUN_STATES = [
  "initializing",
  "running",
  "cancelling",
  "succeeded",
  "failed",
  "cancelled",
] as const satisfies readonly WorkflowRunState[];

export type WorkflowRunState = components["schemas"]["WorkflowRunState"];
export type RunPage = components["schemas"]["RunPage"];
export type RunStatus = components["schemas"]["RunStatus"];
export type RunSummary = components["schemas"]["RunSummary"];
export type StageAttempt = components["schemas"]["StageAttempt"];
export type StageTransition = components["schemas"]["StageTransition"];
export type PlannerPlan = components["schemas"]["PlannerPlan"];
export type CancelRunResponse = components["schemas"]["CancelRunResponse"];
export type ArtifactLineagePage = components["schemas"]["ArtifactLineagePage"];

export interface RunPageRequest {
  state?: WorkflowRunState;
  cursor?: string;
}

export interface RunArtifactPageRequest {
  runId: string;
  namespace?: string;
  cursor?: string;
}

export interface RunArtifactDetailRequest extends ArtifactDetailRequest {
  runId: string;
}

function requireData<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): T {
  if (result.data === undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
  return result.data;
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
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Run response",
  });
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
  return state === "succeeded" || state === "failed" || state === "cancelled";
}

export async function listRuns(
  api: PublicAPI,
  request: RunPageRequest = {},
): Promise<RunPage> {
  const result = await api.request((client) =>
    client.GET("/v1/runs", {
      params: {
        query: {
          limit: RUN_PAGE_SIZE,
          ...(request.state === undefined ? {} : { state: request.state }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
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
  if (run.runId !== runId || !RUN_STATES.includes(run.state)) {
    throw invalidRunResponse(result.response.status);
  }
  if (
    !Array.isArray(run.labels) ||
    run.labels.some(
      (label, index) =>
        typeof label !== "string" ||
        label === "default" ||
        (index > 0 && run.labels[index - 1]! >= label),
    )
  ) {
    throw invalidRunResponse(result.response.status);
  }
  return {
    runId: run.runId,
    workflow: run.workflow,
    state: run.state,
    labels: [...run.labels],
    runtimeConfiguration: safeRunRuntimeConfiguration(run.runtimeConfiguration),
    ...(run.cancellation === undefined
      ? {}
      : { cancellation: run.cancellation }),
    ...(run.parameters === undefined ? {} : { parameters: run.parameters }),
    ...(run.inputs === undefined ? {} : { inputs: run.inputs }),
    attempts: run.attempts.map((attempt) => ({
      ...attempt,
      ...(attempt.runtimeConfiguration === undefined
        ? {}
        : {
            runtimeConfiguration: safeStageRuntimeConfiguration(
              attempt.runtimeConfiguration,
            ),
          }),
    })),
    transitions: [...run.transitions],
    outputs: { ...run.outputs },
    ...(run.eventCursor === undefined ? {} : { eventCursor: run.eventCursor }),
    ...(run.activeStageExecutionId === undefined
      ? {}
      : { activeStageExecutionId: run.activeStageExecutionId }),
    ...(run.createdAt === undefined ? {} : { createdAt: run.createdAt }),
    ...(run.updatedAt === undefined ? {} : { updatedAt: run.updatedAt }),
    ...(run.startedAt === undefined ? {} : { startedAt: run.startedAt }),
    ...(run.finishedAt === undefined ? {} : { finishedAt: run.finishedAt }),
  };
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
  if (
    (result.response.status !== 200 && result.response.status !== 202) ||
    response.runId !== runId ||
    !RUN_STATES.includes(response.state)
  ) {
    throw invalidRunResponse(result.response.status);
  }
  return response;
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
  return requireData(result);
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
