import type { ArtifactArchiveScope } from "./artifact-archive";
import {
  downloadArtifact,
  getArtifactLineage,
  getArtifactMetadata,
  listArtifactVersions,
  previewArtifact,
  writeArtifact,
  type ArtifactDetailRequest,
  type ArtifactLineagePage,
  type ArtifactMetadata,
  type ArtifactPage,
  type ArtifactWriteRequest,
  type ArtifactWriteResponse,
  type DownloadedArtifact,
} from "./artifacts";
import type { PublicAPI } from "./client";
import {
  downloadProjectArtifact,
  getProjectArtifactLineage,
  getProjectArtifactMetadata,
  listProjectArtifactVersions,
  previewProjectArtifact,
  writeProjectArtifact,
} from "./project-artifacts";
import {
  downloadRunArtifact,
  getRunArtifactLineage,
  getRunArtifactMetadata,
  listRunArtifactVersions,
  previewRunArtifact,
} from "./runs";

/** The User, Project or Run scope that owns an Artifact binding. */
export type ArtifactScope = ArtifactArchiveScope;

/** Scopes whose Artifact bindings accept uploads. */
export type WritableArtifactScope = Exclude<ArtifactScope, { kind: "run" }>;

type PageRequest = ArtifactDetailRequest & { cursor?: string };

/** Reads one Artifact binding's revisions through its scope's endpoints. */
export interface ScopedArtifactAPI {
  metadata: (request: ArtifactDetailRequest) => Promise<ArtifactMetadata>;
  versions: (request: PageRequest) => Promise<ArtifactPage>;
  lineage: (request: PageRequest) => Promise<ArtifactLineagePage>;
  download: (metadata: ArtifactMetadata) => Promise<DownloadedArtifact>;
  preview: (metadata: ArtifactMetadata) => Promise<string>;
}

export function scopedArtifactAPI(
  api: PublicAPI,
  scope: ArtifactScope,
): ScopedArtifactAPI {
  switch (scope.kind) {
    case "user":
      return {
        metadata: (request) => getArtifactMetadata(api, request),
        versions: (request) => listArtifactVersions(api, request),
        lineage: (request) => getArtifactLineage(api, request),
        download: (metadata) => downloadArtifact(api, metadata),
        preview: (metadata) => previewArtifact(api, metadata),
      };
    case "project":
      return {
        metadata: (request) =>
          getProjectArtifactMetadata(api, { ...request, projectId: scope.id }),
        versions: (request) =>
          listProjectArtifactVersions(api, { ...request, projectId: scope.id }),
        lineage: (request) =>
          getProjectArtifactLineage(api, { ...request, projectId: scope.id }),
        download: (metadata) =>
          downloadProjectArtifact(api, scope.id, metadata),
        preview: (metadata) => previewProjectArtifact(api, scope.id, metadata),
      };
    case "run":
      return {
        metadata: (request) =>
          getRunArtifactMetadata(api, { ...request, runId: scope.id }),
        versions: (request) =>
          listRunArtifactVersions(api, { ...request, runId: scope.id }),
        lineage: (request) =>
          getRunArtifactLineage(api, { ...request, runId: scope.id }),
        download: (metadata) => downloadRunArtifact(api, scope.id, metadata),
        preview: (metadata) => previewRunArtifact(api, scope.id, metadata),
      };
  }
}

/** Writes one revision to a User or Project Artifact binding. */
export function writeScopeArtifact(
  api: PublicAPI,
  scope: WritableArtifactScope,
  request: ArtifactWriteRequest,
): Promise<ArtifactWriteResponse> {
  return scope.kind === "user"
    ? writeArtifact(api, request)
    : writeProjectArtifact(api, { ...request, projectId: scope.id });
}
