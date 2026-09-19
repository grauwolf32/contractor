import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  MAXIMUM_ARTIFACT_BYTES,
  MAXIMUM_PREVIEW_BYTES,
  type ArtifactMetadata,
  type ExactArtifactRef,
} from "./artifacts";
import type { PublicAPI } from "./client";
import { PublicAPIError } from "./error";
import type { components } from "./generated/public";
import { PROJECT_ID_PATTERN } from "./projects";
import { RUN_ID_PATTERN } from "./runs";

export type ArtifactArchive = components["schemas"]["ArtifactArchive"];
export type ArtifactArchiveEntry =
  components["schemas"]["ArtifactArchiveEntry"];
export type ArtifactArchiveFile = components["schemas"]["ArtifactArchiveFile"];
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

function invalidResponse(): PublicAPIError {
  return new PublicAPIError({
    status: 0,
    code: "invalid_api_response",
    message: "Server returned an invalid archive preview",
  });
}

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function readArchiveJSON(
  api: PublicAPI,
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  signal: AbortSignal,
  path?: string,
): Promise<Record<string, unknown>> {
  const response = await api.fetch(archiveURL(scope, ref, path), {
    method: "GET",
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) throw await api.error(response);
  // 4096 paths of 1024 bytes can expand sixfold under JSON's HTML escaping.
  const maximum = path === undefined ? 32 * 1024 * 1024 : 2 * 1024 * 1024;
  if (
    response.headers.get("ETag") !== `"${ref.revision}"` ||
    response.headers.get("Content-Type")?.split(";")[0]?.trim() !==
      "application/json" ||
    response.body === null
  ) {
    await response.body?.cancel();
    throw invalidResponse();
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > maximum) {
        await reader.cancel();
        throw invalidResponse();
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  let data: unknown;
  try {
    data = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw invalidResponse();
  }
  if (
    !object(data) ||
    !object(data.artifact) ||
    data.artifact.namespace !== ref.namespace ||
    data.artifact.name !== ref.name ||
    data.artifact.revision !== ref.revision
  ) {
    throw invalidResponse();
  }
  return data;
}

export async function getArtifactArchive(
  api: PublicAPI,
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  signal: AbortSignal,
): Promise<ArtifactArchive> {
  const data = await readArchiveJSON(api, scope, ref, signal);
  if (!Array.isArray(data.entries) || data.entries.length > 4096)
    throw invalidResponse();
  const seen = new Map<string, string>();
  for (const entry of data.entries) {
    if (
      !object(entry) ||
      typeof entry.path !== "string" ||
      !validPath(entry.path) ||
      (entry.kind !== "file" && entry.kind !== "directory") ||
      typeof entry.size !== "number" ||
      !Number.isSafeInteger(entry.size) ||
      entry.size < 0 ||
      entry.size > 256 * 1024 * 1024 ||
      typeof entry.previewable !== "boolean" ||
      (entry.previewable &&
        (entry.kind !== "file" || entry.size > MAXIMUM_PREVIEW_BYTES)) ||
      seen.has(entry.path)
    )
      throw invalidResponse();
    seen.set(entry.path, entry.kind);
  }
  for (const path of seen.keys()) {
    const parts = path.split("/");
    for (let i = 1; i < parts.length; i++) {
      if (seen.get(parts.slice(0, i).join("/")) !== "directory")
        throw invalidResponse();
    }
  }
  return data as unknown as ArtifactArchive;
}

export async function getArtifactArchiveFile(
  api: PublicAPI,
  scope: ArtifactArchiveScope,
  ref: ExactArtifactRef,
  path: string,
  signal: AbortSignal,
): Promise<ArtifactArchiveFile> {
  const data = await readArchiveJSON(api, scope, ref, signal, path);
  if (
    data.path !== path ||
    typeof data.text !== "string" ||
    typeof data.size !== "number" ||
    data.size > MAXIMUM_PREVIEW_BYTES ||
    data.size !== new TextEncoder().encode(data.text).length
  ) {
    throw invalidResponse();
  }
  return data as unknown as ArtifactArchiveFile;
}
