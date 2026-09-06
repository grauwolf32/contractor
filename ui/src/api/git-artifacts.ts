import type { PublicAPI } from "./client";
import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  MAXIMUM_ARTIFACT_BYTES,
} from "./artifacts";
import { publicAPIError } from "./error";
import type { components } from "./generated/public";

export type GitKeyState = components["schemas"]["GitKeyState"];
export type GitImportResult = components["schemas"]["GitImportResult"];
export type GitSource = components["schemas"]["GitSource"];
export const gitKeyQueryKey = ["settings", "git-key"] as const;

async function data<T>(result: {
  data?: T;
  error?: unknown;
  response: Response;
}): Promise<T> {
  if (result.data === undefined)
    throw publicAPIError(result.response.status, result.error);
  return result.data;
}
export async function getGitKey(
  api: PublicAPI,
  signal?: AbortSignal,
): Promise<GitKeyState> {
  return data(
    await api.request((client) =>
      client.GET("/v1/settings/git-key", { signal: signal ?? null }),
    ),
  );
}
export async function replaceGitKey(
  api: PublicAPI,
  privateKey: string,
  signal: AbortSignal,
): Promise<GitKeyState> {
  return data(
    await api.request((client) =>
      client.PUT("/v1/settings/git-key", { body: { privateKey }, signal }),
    ),
  );
}
export async function removeGitKey(
  api: PublicAPI,
  signal: AbortSignal,
): Promise<void> {
  const result = await api.request((client) =>
    client.DELETE("/v1/settings/git-key", { signal }),
  );
  if (!result.response.ok)
    throw publicAPIError(result.response.status, result.error);
}
export interface GitImportDraft {
  projectId?: string;
  namespace: string;
  name: string;
  repositoryUrl: string;
  ref?: string;
  expectedRevision?: string;
}
export async function importGitArtifact(
  api: PublicAPI,
  draft: GitImportDraft,
  signal: AbortSignal,
): Promise<GitImportResult> {
  if (
    !ARTIFACT_NAME_PATTERN.test(draft.namespace) ||
    !ARTIFACT_NAME_PATTERN.test(draft.name)
  )
    throw new TypeError(
      "Artifact namespace and name must use ASCII letters, digits, dots, underscores or hyphens, without spaces.",
    );
  const params = {
    path: { namespace: draft.namespace, name: draft.name },
    header:
      draft.expectedRevision === undefined
        ? { "If-None-Match": "*" as const }
        : { "If-Match": `"${draft.expectedRevision}"` },
  };
  const body = {
    repositoryUrl: draft.repositoryUrl,
    ...(draft.ref === undefined ? {} : { ref: draft.ref }),
  };
  const response =
    draft.projectId === undefined
      ? await api.request((client) =>
          client.POST("/v1/artifacts/{namespace}/{name}/git-import", {
            params,
            body,
            signal,
          }),
        )
      : await api.request((client) =>
          client.POST(
            "/v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import",
            {
              params: {
                ...params,
                path: { ...params.path, projectId: draft.projectId! },
              },
              body,
              signal,
            },
          ),
        );
  const result = await data(response);
  if (
    result.artifact?.namespace !== draft.namespace ||
    result.artifact.name !== draft.name ||
    !ARTIFACT_REVISION_PATTERN.test(result.artifact.revision) ||
    result.mediaType !== "application/zip" ||
    !Number.isSafeInteger(result.size) ||
    result.size <= 0 ||
    result.size > MAXIMUM_ARTIFACT_BYTES ||
    !/^[a-f0-9]{40}$/.test(result.gitSource?.resolvedCommit ?? "")
  ) {
    throw publicAPIError(0, {
      code: "invalid_api_response",
      message:
        "Git import response is incomplete; inspect the Artifact library before retrying",
      retryable: false,
    });
  }
  return result;
}
