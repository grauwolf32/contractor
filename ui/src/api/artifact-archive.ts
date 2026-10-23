import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  MAXIMUM_ARTIFACT_BYTES,
  type ArtifactMetadata,
  type ExactArtifactRef,
} from "./artifacts";
import type { PublicAPI } from "./client";
import { schemaLimit, type ResponseOperation } from "./contract";
import { invalidAPIResponse, PublicAPIError } from "./error";
import type { components } from "./generated/public";
import { PROJECT_ID_PATTERN } from "./projects";
import { RUN_ID_PATTERN } from "./runs";

type ArtifactArchive = components["schemas"]["ArtifactArchive"];
export type ArtifactArchiveEntry =
  components["schemas"]["ArtifactArchiveEntry"];
type ArtifactArchiveFile = components["schemas"]["ArtifactArchiveFile"];
// The Server's text bound for one archive file, read from its schema.
const MAXIMUM_ARCHIVE_TEXT_BYTES = schemaLimit(
  "ArtifactArchiveFile",
  "maximum",
  "properties",
  "size",
);
export const ARCHIVE_TEXT_LIMIT_LABEL = `${MAXIMUM_ARCHIVE_TEXT_BYTES / (1024 * 1024)} MiB`;

export type ArtifactArchiveScope =
  | { kind: "user" }
  | { kind: "project"; id: string }
  | { kind: "run"; id: string };

export function canBrowseArchive(metadata: ArtifactMetadata): boolean {
  return (
    metadata.size <= MAXIMUM_ARTIFACT_BYTES &&
    [
      "application/zip",
      "application/x-zip-compressed",
      "application/vnd.contractor.agent-skill+zip",
    ].includes(metadata.mediaType)
  );
}

export function archiveQueryKey(
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
) {
  return [
    "artifact-archive",
    scope.kind,
    "id" in scope ? scope.id : null,
    ref.namespace,
    ref.name,
    ref.revision,
  ] as const;
}

function validPath(path: string): boolean {
  const parts = path.split("/");
  return (
    path !== "" &&
    new TextEncoder().encode(path).length <= 1024 &&
    !/[\\:\p{Cc}\p{Cf}]/u.test(path) &&
    parts.length <= 32 &&
    parts.every((part) => part !== "" && part !== "." && part !== "..")
  );
}

function archiveURL(
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  path?: string,
): string {
  if (
    !ARTIFACT_NAME_PATTERN.test(ref.namespace) ||
    !ARTIFACT_NAME_PATTERN.test(ref.name) ||
    !ARTIFACT_REVISION_PATTERN.test(ref.revision)
  ) {
    throw new TypeError("Archive requires an exact Artifact reference");
  }
  let prefix = "/v1";
  if (scope.kind !== "user") {
    const pattern =
      scope.kind === "project" ? PROJECT_ID_PATTERN : RUN_ID_PATTERN;
    if (!pattern.test(scope.id))
      throw new TypeError("Archive scope is invalid");
    prefix += `/${scope.kind === "project" ? "projects" : "runs"}/${encodeURIComponent(scope.id)}`;
  }
  const query = new URLSearchParams({ revision: ref.revision });
  if (path !== undefined) {
    if (!validPath(path)) throw new TypeError("Archive file path is invalid");
    query.set("path", path);
  }
  return `${prefix}/artifacts/${encodeURIComponent(ref.namespace)}/${encodeURIComponent(ref.name)}/archive${path === undefined ? "" : "/file"}?${query}`;
}

function invalidResponse(status: number): PublicAPIError {
  return invalidAPIResponse(
    status,
    "Server returned an invalid archive preview",
  );
}

/** The operation `archiveURL` addresses for this scope. */
function archiveOperation(
  scope: ArtifactArchiveScope,
  file: boolean,
): ResponseOperation {
  const suffix = file ? "/archive/file" : "/archive";
  switch (scope.kind) {
    case "user":
      return `GET /v1/artifacts/{namespace}/{name}${suffix}`;
    case "project":
      return `GET /v1/projects/{projectId}/artifacts/{namespace}/{name}${suffix}`;
    case "run":
      return `GET /v1/runs/{runId}/artifacts/{namespace}/{name}${suffix}`;
  }
}

async function readArchiveJSON<T extends { artifact: ExactArtifactRef }>(
  api: PublicAPI,
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  signal: AbortSignal,
  path?: string,
): Promise<T> {
  const response = await api.fetch(archiveURL(scope, ref, path), {
    method: "GET",
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    const error = await api.error(response);
    if (error.code === "archive_preview_limit") {
      throw new PublicAPIError({
        status: error.status,
        code: error.code,
        message:
          path === undefined
            ? "This archive exceeds the browsing limits for its entries or expanded content. Download the original archive to inspect it."
            : `This file exceeds the ${ARCHIVE_TEXT_LIMIT_LABEL} text preview limit. Download the original archive to inspect it.`,
        ...(error.requestId === undefined
          ? {}
          : { requestId: error.requestId }),
      });
    }
    throw error;
  }
  // Cross-field: the ETag header names the requested revision.
  if (response.headers.get("ETag") !== `"${ref.revision}"`) {
    await response.body?.cancel();
    throw invalidResponse(response.status);
  }
  // 4096 paths of 1024 bytes, or one file's text, can expand sixfold under
  // JSON's HTML escaping; 64 KiB covers the rest of a file response.
  const data = await api.readJSON<T>(
    archiveOperation(scope, path !== undefined),
    response,
    path === undefined
      ? 32 * 1024 * 1024
      : 6 * MAXIMUM_ARCHIVE_TEXT_BYTES + 64 * 1024,
  );
  // Cross-field: the Server answered for the requested Artifact revision.
  if (
    data.artifact.namespace !== ref.namespace ||
    data.artifact.name !== ref.name ||
    data.artifact.revision !== ref.revision
  ) {
    throw invalidResponse(response.status);
  }
  return data;
}

export async function getArtifactArchive(
  api: PublicAPI,
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  signal: AbortSignal,
): Promise<ArtifactArchive> {
  const data = await readArchiveJSON<ArtifactArchive>(api, scope, ref, signal);
  // Cross-field: paths are unique, every parent is a directory entry, and
  // only a file within the preview limit is previewable.
  const kinds = new Map<string, ArtifactArchiveEntry["kind"]>();
  for (const entry of data.entries) {
    if (
      kinds.has(entry.path) ||
      (entry.previewable &&
        (entry.kind !== "file" || entry.size > MAXIMUM_ARCHIVE_TEXT_BYTES))
    )
      throw invalidResponse(200);
    kinds.set(entry.path, entry.kind);
  }
  for (const path of kinds.keys()) {
    const parts = path.split("/");
    for (let i = 1; i < parts.length; i++) {
      if (kinds.get(parts.slice(0, i).join("/")) !== "directory")
        throw invalidResponse(200);
    }
  }
  return data;
}

export async function getArtifactArchiveFile(
  api: PublicAPI,
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  path: string,
  signal: AbortSignal,
): Promise<ArtifactArchiveFile> {
  const data = await readArchiveJSON<ArtifactArchiveFile>(
    api,
    scope,
    ref,
    signal,
    path,
  );
  // Cross-field: the requested file, whose size is its text's UTF-8 length.
  if (
    data.path !== path ||
    data.size !== new TextEncoder().encode(data.text).length
  ) {
    throw invalidResponse(200);
  }
  return data;
}
