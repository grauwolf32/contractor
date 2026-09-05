import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import { safeRunMetadataLabels } from "./run-metadata-labels";
import { RUN_PAGE_SIZE, type RunPage } from "./runs";

export const PROJECT_PAGE_SIZE = 50;
export const PROJECT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
export const PROJECT_REVISION_PATTERN = /^[1-9][0-9]{0,18}$/;
export const MAXIMUM_PROJECT_NAME_LENGTH = 160;
export const MAXIMUM_PROJECT_DESCRIPTION_LENGTH = 4096;
export const MAXIMUM_PROJECT_TARGET_URL_BYTES = 2048;
const RUNTIME_CREDENTIAL_ID_PATTERN = /^[a-z][a-z0-9_-]{0,127}$/;

export type Project = components["schemas"]["Project"];
export type ProjectKind = components["schemas"]["ProjectKind"];
export type ProjectLifecycle = components["schemas"]["ProjectLifecycle"];
export type ProjectDeletionPhase =
  components["schemas"]["ProjectDeletionPhase"];
export type ProjectPage = components["schemas"]["ProjectPage"];
export type CreateProjectRequest =
  components["schemas"]["CreateProjectRequest"];
export type UpdateProjectRequest =
  components["schemas"]["UpdateProjectRequest"];
export type ProjectHTTPTarget = components["schemas"]["ProjectHTTPTarget"];

export interface ProjectPageRequest {
  kind?: ProjectKind;
  cursor?: string;
}

export interface CreateProjectOptions {
  request: CreateProjectRequest;
  idempotencyKey: string;
}

export interface UpdateProjectOptions {
  projectId: string;
  request: UpdateProjectRequest;
  expectedRevision: string;
}

export interface DeleteProjectOptions {
  projectId: string;
  expectedRevision: string;
}

export interface ProjectRunPageRequest {
  projectId: string;
  cursor?: string;
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

function invalidProjectResponse(status: number): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Project response",
  });
}

export function requireProjectID(projectId: string): void {
  if (!PROJECT_ID_PATTERN.test(projectId)) {
    throw new TypeError("Project ID is invalid");
  }
}

function requireProjectRevision(revision: string): void {
  if (!PROJECT_REVISION_PATTERN.test(revision)) {
    throw new TypeError("Project revision is invalid");
  }
}

export function normalizeProjectRequest(
  request: CreateProjectRequest,
): CreateProjectRequest {
  const name = request.name.trim();
  const description = request.description?.trim() ?? "";
  if (
    (request.kind !== "project" && request.kind !== "evaluation") ||
    name.length === 0 ||
    name.length > MAXIMUM_PROJECT_NAME_LENGTH ||
    description.length > MAXIMUM_PROJECT_DESCRIPTION_LENGTH
  ) {
    throw new TypeError("Project metadata is invalid");
  }
  return {
    kind: request.kind,
    name,
    ...(description === "" ? {} : { description }),
  };
}

function safeProject(value: Project, status: number): Project {
  const deletionValid =
    value.deletion !== undefined &&
    (value.deletion.phase === "cancelling" ||
      value.deletion.phase === "draining" ||
      value.deletion.phase === "purging_runs" ||
      value.deletion.phase === "purging_artifacts") &&
    typeof value.deletion.requestedAt === "string" &&
    value.deletion.requestedAt.length > 0;
  if (
    !PROJECT_ID_PATTERN.test(value.projectId) ||
    (value.kind !== "project" && value.kind !== "evaluation") ||
    typeof value.name !== "string" ||
    value.name.length === 0 ||
    value.name.length > MAXIMUM_PROJECT_NAME_LENGTH ||
    typeof value.description !== "string" ||
    value.description.length > MAXIMUM_PROJECT_DESCRIPTION_LENGTH ||
    (value.lifecycle !== "active" && value.lifecycle !== "deleting") ||
    (value.lifecycle === "active"
      ? value.deletion !== undefined
      : !deletionValid) ||
    !PROJECT_REVISION_PATTERN.test(value.revision) ||
    typeof value.createdAt !== "string" ||
    typeof value.updatedAt !== "string"
  ) {
    throw invalidProjectResponse(status);
  }
  return {
    ...value,
    ...(value.httpTarget === undefined
      ? {}
      : { httpTarget: normalizeProjectHTTPTarget(value.httpTarget) }),
  };
}

export function normalizeProjectHTTPTarget(
  target: ProjectHTTPTarget,
): ProjectHTTPTarget {
  const url = target.url.trim();
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new TypeError("Project HTTP target URL is invalid");
  }
  if (
    new TextEncoder().encode(url).length > MAXIMUM_PROJECT_TARGET_URL_BYTES ||
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.search !== "" ||
    parsed.hash !== ""
  ) {
    throw new TypeError("Project HTTP target URL is invalid");
  }
  const credential = target.credential;
  if (
    credential !== undefined &&
    (!RUNTIME_CREDENTIAL_ID_PATTERN.test(credential.credentialId) ||
      (credential.kind !== "http-origin-basic@1" &&
        credential.kind !== "http-origin-bearer@1"))
  ) {
    throw new TypeError("Project HTTP target credential is invalid");
  }
  return {
    url,
    ...(credential === undefined ? {} : { credential: { ...credential } }),
  };
}

function requireETag(response: Response, revision: string): void {
  if (response.headers.get("ETag") !== `"${revision}"`) {
    throw invalidProjectResponse(response.status);
  }
}

export async function listProjects(
  api: PublicAPI,
  request: ProjectPageRequest = {},
): Promise<ProjectPage> {
  const result = await api.request((client) =>
    client.GET("/v1/projects", {
      params: {
        query: {
          limit: PROJECT_PAGE_SIZE,
          ...(request.kind === undefined ? {} : { kind: request.kind }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  if (!Array.isArray(page.items) || page.page === undefined) {
    throw invalidProjectResponse(result.response.status);
  }
  return {
    items: page.items.map((project) =>
      safeProject(project, result.response.status),
    ),
    page: { ...page.page },
  };
}

export async function getProject(
  api: PublicAPI,
  projectId: string,
): Promise<Project> {
  requireProjectID(projectId);
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}", {
      params: { path: { projectId } },
    }),
  );
  const project = safeProject(requireData(result), result.response.status);
  if (project.projectId !== projectId) {
    throw invalidProjectResponse(result.response.status);
  }
  requireETag(result.response, project.revision);
  return project;
}

export async function createProject(
  api: PublicAPI,
  options: CreateProjectOptions,
): Promise<Project> {
  const request = normalizeProjectRequest(options.request);
  const result = await api.request((client) =>
    client.POST("/v1/projects", {
      params: { header: { "Idempotency-Key": options.idempotencyKey } },
      body: request,
    }),
  );
  const project = safeProject(requireData(result), result.response.status);
  if (project.kind !== request.kind) {
    throw invalidProjectResponse(result.response.status);
  }
  requireETag(result.response, project.revision);
  return project;
}

export async function updateProject(
  api: PublicAPI,
  options: UpdateProjectOptions,
): Promise<Project> {
  requireProjectID(options.projectId);
  requireProjectRevision(options.expectedRevision);
  if (Object.keys(options.request).length === 0) {
    throw new TypeError("Project update must change at least one field");
  }
  const name = options.request.name?.trim();
  const description = options.request.description?.trim();
  const targetPresent = Object.hasOwn(options.request, "httpTarget");
  const httpTarget = options.request.httpTarget;
  if (
    (name !== undefined &&
      (name.length === 0 || name.length > MAXIMUM_PROJECT_NAME_LENGTH)) ||
    (description !== undefined &&
      description.length > MAXIMUM_PROJECT_DESCRIPTION_LENGTH)
  ) {
    throw new TypeError("Project metadata is invalid");
  }
  const body: UpdateProjectRequest = {
    ...(name === undefined ? {} : { name }),
    ...(description === undefined ? {} : { description }),
    ...(targetPresent
      ? {
          httpTarget:
            httpTarget === null || httpTarget === undefined
              ? null
              : normalizeProjectHTTPTarget(httpTarget),
        }
      : {}),
  };
  const result = await api.request((client) =>
    client.PATCH("/v1/projects/{projectId}", {
      params: {
        path: { projectId: options.projectId },
        header: { "If-Match": `"${options.expectedRevision}"` },
      },
      body,
    }),
  );
  const project = safeProject(requireData(result), result.response.status);
  if (project.projectId !== options.projectId) {
    throw invalidProjectResponse(result.response.status);
  }
  requireETag(result.response, project.revision);
  return project;
}

export async function deleteProject(
  api: PublicAPI,
  options: DeleteProjectOptions,
): Promise<Project> {
  requireProjectID(options.projectId);
  requireProjectRevision(options.expectedRevision);
  const result = await api.request((client) =>
    client.DELETE("/v1/projects/{projectId}", {
      params: {
        path: { projectId: options.projectId },
        header: { "If-Match": `"${options.expectedRevision}"` },
      },
    }),
  );
  const project = safeProject(requireData(result), result.response.status);
  if (
    project.projectId !== options.projectId ||
    project.lifecycle !== "deleting"
  ) {
    throw invalidProjectResponse(result.response.status);
  }
  requireETag(result.response, project.revision);
  return project;
}

export async function listProjectRuns(
  api: PublicAPI,
  request: ProjectRunPageRequest,
): Promise<RunPage> {
  requireProjectID(request.projectId);
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}/runs", {
      params: {
        path: { projectId: request.projectId },
        query: {
          limit: RUN_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  if (!Array.isArray(page.items) || page.page === undefined) {
    throw invalidProjectResponse(result.response.status);
  }
  return {
    items: page.items.map((run) => ({
      ...run,
      labels: safeRunMetadataLabels(run.labels ?? {}),
    })),
    page: { ...page.page },
  };
}
