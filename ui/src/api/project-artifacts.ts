import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_PAGE_SIZE,
  ARTIFACT_REVISION_PATTERN,
  downloadExactArtifact,
  previewExactArtifact,
  writeScopedArtifact,
  type ArtifactDetailRequest,
  type ArtifactLineagePage,
  type ArtifactMetadata,
  type ArtifactPage,
  type ArtifactPageRequest,
  type ArtifactWriteRequest,
  type ArtifactWriteResponse,
  type DownloadedArtifact,
} from "./artifacts";
import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import { requireProjectID } from "./projects";

export interface ProjectArtifactPageRequest extends ArtifactPageRequest {
  projectId: string;
}

export interface ProjectArtifactDetailRequest extends ArtifactDetailRequest {
  projectId: string;
}

export interface ProjectArtifactWriteRequest extends ArtifactWriteRequest {
  projectId: string;
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

function invalidProjectArtifactResponse(status: number): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Project Artifact response",
  });
}

function requireArtifactIdentity(namespace: string, name: string): void {
  if (
    !ARTIFACT_NAME_PATTERN.test(namespace) ||
    !ARTIFACT_NAME_PATTERN.test(name)
  ) {
    throw new TypeError("Project Artifact identity is invalid");
  }
}

function requireRevision(revision: string): void {
  if (!ARTIFACT_REVISION_PATTERN.test(revision)) {
    throw new TypeError("Project Artifact revision is invalid");
  }
}

function projectArtifactPath(
  projectId: string,
  namespace: string,
  name: string,
): string {
  requireProjectID(projectId);
  requireArtifactIdentity(namespace, name);
  return `/v1/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
}

function exactRevisionQuery(revision: string): string {
  requireRevision(revision);
  return `?revision=${encodeURIComponent(revision)}`;
}

function safeArtifactPage(page: ArtifactPage, status: number): ArtifactPage {
  if (!Array.isArray(page.items) || page.page === undefined) {
    throw invalidProjectArtifactResponse(status);
  }
  return { items: [...page.items], page: { ...page.page } };
}

export async function listProjectArtifacts(
  api: PublicAPI,
  request: ProjectArtifactPageRequest,
): Promise<ArtifactPage> {
  requireProjectID(request.projectId);
  if (
    request.namespace !== undefined &&
    !ARTIFACT_NAME_PATTERN.test(request.namespace)
  ) {
    throw new TypeError("Project Artifact namespace is invalid");
  }
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}/artifacts", {
      params: {
        path: { projectId: request.projectId },
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
  return safeArtifactPage(requireData(result), result.response.status);
}

export async function getProjectArtifactMetadata(
  api: PublicAPI,
  request: ProjectArtifactDetailRequest,
): Promise<ArtifactMetadata> {
  requireProjectID(request.projectId);
  requireArtifactIdentity(request.namespace, request.name);
  if (request.revision !== undefined) {
    requireRevision(request.revision);
  }
  const result = await api.request((client) =>
    client.GET(
      "/v1/projects/{projectId}/artifacts/{namespace}/{name}/metadata",
      {
        params: {
          path: {
            projectId: request.projectId,
            namespace: request.namespace,
            name: request.name,
          },
          query:
            request.revision === undefined
              ? {}
              : { revision: request.revision },
        },
      },
    ),
  );
  const metadata = requireData(result);
  if (
    metadata.artifact.namespace !== request.namespace ||
    metadata.artifact.name !== request.name ||
    (request.revision !== undefined &&
      metadata.artifact.revision !== request.revision)
  ) {
    throw invalidProjectArtifactResponse(result.response.status);
  }
  return metadata;
}

export async function listProjectArtifactVersions(
  api: PublicAPI,
  request: ProjectArtifactDetailRequest & { cursor?: string },
): Promise<ArtifactPage> {
  requireProjectID(request.projectId);
  requireArtifactIdentity(request.namespace, request.name);
  const result = await api.request((client) =>
    client.GET(
      "/v1/projects/{projectId}/artifacts/{namespace}/{name}/versions",
      {
        params: {
          path: {
            projectId: request.projectId,
            namespace: request.namespace,
            name: request.name,
          },
          query: {
            limit: ARTIFACT_PAGE_SIZE,
            ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
          },
        },
      },
    ),
  );
  return safeArtifactPage(requireData(result), result.response.status);
}

export async function getProjectArtifactLineage(
  api: PublicAPI,
  request: ProjectArtifactDetailRequest & { cursor?: string },
): Promise<ArtifactLineagePage> {
  requireProjectID(request.projectId);
  requireArtifactIdentity(request.namespace, request.name);
  if (request.revision !== undefined) {
    requireRevision(request.revision);
  }
  const result = await api.request((client) =>
    client.GET(
      "/v1/projects/{projectId}/artifacts/{namespace}/{name}/lineage",
      {
        params: {
          path: {
            projectId: request.projectId,
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
      },
    ),
  );
  const page = requireData(result);
  if (!Array.isArray(page.items) || page.page === undefined) {
    throw invalidProjectArtifactResponse(result.response.status);
  }
  return { items: [...page.items], page: { ...page.page } };
}

export function writeProjectArtifact(
  api: PublicAPI,
  request: ProjectArtifactWriteRequest,
): Promise<ArtifactWriteResponse> {
  const path = projectArtifactPath(
    request.projectId,
    request.namespace,
    request.name,
  );
  return writeScopedArtifact(api, path, request);
}

export function downloadProjectArtifact(
  api: PublicAPI,
  projectId: string,
  metadata: ArtifactMetadata,
): Promise<DownloadedArtifact> {
  const path = `${projectArtifactPath(projectId, metadata.artifact.namespace, metadata.artifact.name)}${exactRevisionQuery(metadata.artifact.revision)}`;
  return downloadExactArtifact(api, metadata, path);
}

export function previewProjectArtifact(
  api: PublicAPI,
  projectId: string,
  metadata: ArtifactMetadata,
): Promise<string> {
  return previewExactArtifact(metadata, () =>
    downloadProjectArtifact(api, projectId, metadata),
  );
}
