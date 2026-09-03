import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";

export const MAXIMUM_ARTIFACT_BYTES = 16 * 1024 * 1024;
export const MAXIMUM_PREVIEW_BYTES = 256 * 1024;
export const ARTIFACT_PAGE_SIZE = 50;

export const ARTIFACT_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
export const ARTIFACT_REVISION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
export const MEDIA_TYPE_PATTERN = /^[a-z0-9!#$&^_.+-]+\/[a-z0-9!#$&^_.+-]+$/;

const PREVIEW_MEDIA_TYPES = new Set([
  "application/json",
  "application/vnd.oai.openapi",
  "application/vnd.oai.openapi+json",
  "application/vnd.oai.openapi+yaml",
  "application/yaml",
  "application/x-yaml",
  "text/markdown",
  "text/plain",
  "text/vnd.likec4",
  "text/x-markdown",
  "text/yaml",
]);

const EXTENSIONS = new Map([
  ["application/json", ".json"],
  ["application/vnd.oai.openapi", ".yaml"],
  ["application/vnd.oai.openapi+json", ".json"],
  ["application/vnd.oai.openapi+yaml", ".yaml"],
  ["application/yaml", ".yaml"],
  ["application/x-yaml", ".yaml"],
  ["application/zip", ".zip"],
  ["text/markdown", ".md"],
  ["text/plain", ".txt"],
  ["text/vnd.likec4", ".c4"],
  ["text/x-markdown", ".md"],
  ["text/yaml", ".yaml"],
]);

export type ArtifactMetadata = components["schemas"]["ArtifactMetadata"];
export type ArtifactPage = components["schemas"]["ArtifactPage"];
export type ArtifactLineagePage = components["schemas"]["ArtifactLineagePage"];
export type ArtifactWriteResponse =
  components["schemas"]["ArtifactWriteResponse"];
export type ExactArtifactRef = components["schemas"]["ExactArtifactRef"];

export interface ArtifactPageRequest {
  namespace?: string;
  cursor?: string;
}

export interface ArtifactDetailRequest {
  namespace: string;
  name: string;
  revision?: string;
}

export interface ArtifactWriteRequest {
  namespace: string;
  name: string;
  mediaType: string;
  payload: Blob;
  expectedRevision?: string;
}

export interface DownloadedArtifact {
  blob: Blob;
  mediaType: string;
  filename: string;
}

function artifactPath(namespace: string, name: string): string {
  requireArtifactName("namespace", namespace);
  requireArtifactName("name", name);
  return `/v1/artifacts/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}`;
}

function exactQuery(revision: string): string {
  if (!ARTIFACT_REVISION_PATTERN.test(revision)) {
    throw new TypeError("Artifact revision is invalid");
  }
  return `?revision=${encodeURIComponent(revision)}`;
}

function requireArtifactName(field: string, value: string): void {
  if (!ARTIFACT_NAME_PATTERN.test(value)) {
    throw new TypeError(`Artifact ${field} is invalid`);
  }
}

function requireMediaType(value: string): void {
  if (value !== "*/*" && !MEDIA_TYPE_PATTERN.test(value)) {
    throw new TypeError("Artifact media type is invalid");
  }
}

function requireRevision(value: string): void {
  if (!ARTIFACT_REVISION_PATTERN.test(value)) {
    throw new TypeError("Artifact revision is invalid");
  }
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

function invalidWriteResponse(status = 0): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Artifact write response",
  });
}

function parseWriteResponse(value: unknown): ArtifactWriteResponse {
  if (
    typeof value !== "object" ||
    value === null ||
    !("artifact" in value) ||
    !("mediaType" in value) ||
    !("size" in value)
  ) {
    throw invalidWriteResponse();
  }
  const candidate = value as {
    artifact?: { namespace?: unknown; name?: unknown; revision?: unknown };
    mediaType?: unknown;
    size?: unknown;
  };
  if (
    typeof candidate.artifact?.namespace !== "string" ||
    typeof candidate.artifact.name !== "string" ||
    typeof candidate.artifact.revision !== "string" ||
    typeof candidate.mediaType !== "string" ||
    typeof candidate.size !== "number" ||
    !Number.isSafeInteger(candidate.size) ||
    candidate.size < 0 ||
    candidate.size > MAXIMUM_ARTIFACT_BYTES
  ) {
    throw invalidWriteResponse();
  }
  try {
    requireArtifactName("namespace", candidate.artifact.namespace);
    requireArtifactName("name", candidate.artifact.name);
    requireRevision(candidate.artifact.revision);
    requireMediaType(candidate.mediaType);
  } catch {
    throw invalidWriteResponse();
  }
  return candidate as ArtifactWriteResponse;
}

export async function listArtifacts(
  api: PublicAPI,
  request: ArtifactPageRequest = {},
): Promise<ArtifactPage> {
  const result = await api.request((client) =>
    client.GET("/v1/artifacts", {
      params: {
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

export async function getArtifactMetadata(
  api: PublicAPI,
  request: ArtifactDetailRequest,
): Promise<ArtifactMetadata> {
  const result = await api.request((client) =>
    client.GET("/v1/artifacts/{namespace}/{name}/metadata", {
      params: {
        path: { namespace: request.namespace, name: request.name },
        query:
          request.revision === undefined ? {} : { revision: request.revision },
      },
    }),
  );
  return requireData(result);
}

export async function listArtifactVersions(
  api: PublicAPI,
  request: ArtifactDetailRequest & { cursor?: string },
): Promise<ArtifactPage> {
  const result = await api.request((client) =>
    client.GET("/v1/artifacts/{namespace}/{name}/versions", {
      params: {
        path: { namespace: request.namespace, name: request.name },
        query: {
          limit: ARTIFACT_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function getArtifactLineage(
  api: PublicAPI,
  request: ArtifactDetailRequest & { cursor?: string },
): Promise<ArtifactLineagePage> {
  const result = await api.request((client) =>
    client.GET("/v1/artifacts/{namespace}/{name}/lineage", {
      params: {
        path: { namespace: request.namespace, name: request.name },
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

export async function writeArtifact(
  api: PublicAPI,
  request: ArtifactWriteRequest,
): Promise<ArtifactWriteResponse> {
  const path = artifactPath(request.namespace, request.name);
  requireMediaType(request.mediaType);
  if (request.mediaType === "*/*") {
    throw new TypeError("Artifact upload requires one concrete media type");
  }
  if (request.payload.size > MAXIMUM_ARTIFACT_BYTES) {
    throw new TypeError("Artifact payload exceeds 16 MiB");
  }
  let headers: Headers;
  if (request.expectedRevision === undefined) {
    headers = api.mutationHeaders({ ifNoneMatch: "*" });
  } else {
    requireRevision(request.expectedRevision);
    headers = api.mutationHeaders({
      ifMatch: `"${request.expectedRevision}"`,
    });
  }
  headers.set("Content-Type", request.mediaType);
  // Materialize the bounded Blob before crossing the Request boundary. Besides
  // making the exact bytes explicit, this avoids cross-realm Blob ambiguity
  // when the UI and Fetch implementation come from different browser realms.
  const payload = await request.payload.arrayBuffer();
  const response = await api.fetch(path, {
    method: "PUT",
    headers,
    body: payload,
  });
  if (!response.ok) {
    throw await api.error(response);
  }
  let value: unknown;
  try {
    value = await response.json();
  } catch {
    throw new PublicAPIError({
      status: response.status,
      code: "invalid_api_response",
      message: "Server returned an unreadable Artifact write response",
    });
  }
  const result = parseWriteResponse(value);
  if (
    result.artifact.namespace !== request.namespace ||
    result.artifact.name !== request.name ||
    result.mediaType !== request.mediaType ||
    result.size !== request.payload.size ||
    response.headers.get("ETag") !== `"${result.artifact.revision}"`
  ) {
    throw invalidWriteResponse(response.status);
  }
  return result;
}

export function canPreviewArtifact(metadata: ArtifactMetadata): boolean {
  return (
    metadata.size <= MAXIMUM_PREVIEW_BYTES &&
    PREVIEW_MEDIA_TYPES.has(metadata.mediaType)
  );
}

export function suggestedArtifactFilename(
  artifact: ExactArtifactRef,
  mediaType: string,
): string {
  requireArtifactName("namespace", artifact.namespace);
  requireArtifactName("name", artifact.name);
  requireRevision(artifact.revision);
  const safeRevision = artifact.revision.replace(/[^A-Za-z0-9_.-]/g, "_");
  return `${artifact.namespace}-${artifact.name}-${safeRevision}${EXTENSIONS.get(mediaType) ?? ".bin"}`;
}

export async function downloadArtifact(
  api: PublicAPI,
  metadata: ArtifactMetadata,
): Promise<DownloadedArtifact> {
  const path = `${artifactPath(metadata.artifact.namespace, metadata.artifact.name)}${exactQuery(metadata.artifact.revision)}`;
  return downloadExactArtifact(api, metadata, path);
}

export async function downloadExactArtifact(
  api: PublicAPI,
  metadata: ArtifactMetadata,
  path: string,
): Promise<DownloadedArtifact> {
  const response = await api.fetch(path, {
    method: "GET",
    headers: { Accept: metadata.mediaType },
  });
  if (!response.ok) {
    throw await api.error(response);
  }
  const declaredLength = response.headers.get("Content-Length");
  if (
    declaredLength !== null &&
    (!/^(?:0|[1-9][0-9]*)$/.test(declaredLength) ||
      Number(declaredLength) > MAXIMUM_ARTIFACT_BYTES)
  ) {
    throw new PublicAPIError({
      status: response.status,
      code: "invalid_api_response",
      message: "Server returned an invalid Artifact length",
    });
  }
  const mediaType = response.headers.get("Content-Type") ?? "";
  if (mediaType !== metadata.mediaType) {
    throw new PublicAPIError({
      status: response.status,
      code: "invalid_api_response",
      message: "Artifact media type changed during download",
    });
  }
  const blob = await response.blob();
  if (blob.size > MAXIMUM_ARTIFACT_BYTES || blob.size !== metadata.size) {
    throw new PublicAPIError({
      status: response.status,
      code: "invalid_api_response",
      message: "Artifact bytes do not match authoritative metadata",
    });
  }
  return {
    blob,
    mediaType,
    filename: suggestedArtifactFilename(metadata.artifact, mediaType),
  };
}

export async function previewArtifact(
  api: PublicAPI,
  metadata: ArtifactMetadata,
): Promise<string> {
  return previewExactArtifact(metadata, () => downloadArtifact(api, metadata));
}

export async function previewExactArtifact(
  metadata: ArtifactMetadata,
  download: () => Promise<DownloadedArtifact>,
): Promise<string> {
  if (!canPreviewArtifact(metadata)) {
    throw new TypeError(
      "This Artifact is not eligible for bounded text preview",
    );
  }
  const downloaded = await download();
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(
      await downloaded.blob.arrayBuffer(),
    );
  } catch {
    throw new PublicAPIError({
      status: 0,
      code: "invalid_artifact_text",
      message: "Artifact is not valid UTF-8 text",
    });
  }
}
