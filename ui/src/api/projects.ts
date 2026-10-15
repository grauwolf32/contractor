import type { PublicAPI } from "./client";
import { invalidAPIResponse, PublicAPIError, requireData } from "./error";
import type { components } from "./generated/public";
import {
  RUN_PAGE_SIZE,
  RUN_STATES,
  type RunPage,
  type WorkflowRunState,
  type WorkflowRunLifecycle,
} from "./runs";

export const PROJECT_PAGE_SIZE = 50;
export const PROJECT_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
export const PROJECT_REVISION_PATTERN = /^[1-9][0-9]{0,18}$/;
export const MAXIMUM_PROJECT_NAME_LENGTH = 160;
export const MAXIMUM_PROJECT_DESCRIPTION_LENGTH = 4096;
export const MAXIMUM_PROJECT_TARGET_URL_BYTES = 2048;
const RUNTIME_CREDENTIAL_ID_PATTERN = /^[a-z][a-z0-9_-]{0,127}$/;

export type Project = components["schemas"]["Project"];
export type ActiveChecksPolicy = components["schemas"]["ActiveChecksPolicy"];

export function requireActiveChecksPolicy(value: unknown): ActiveChecksPolicy {
  if (
    value === "automatic" ||
    value === "approval-required" ||
    value === "prohibited"
  )
    return value;
  throw new TypeError("Project active checks policy is invalid");
}
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
  limit?: number;
  state?: WorkflowRunState;
  lifecycle?: WorkflowRunLifecycle;
}

function invalidProjectResponse(status: number): PublicAPIError {
  return invalidAPIResponse(
    status,
    "Server returned an invalid Project response",
  );
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
    activeChecks: requireActiveChecksPolicy(
      request.activeChecks === undefined ? "automatic" : request.activeChecks,
    ),
    ...(description === "" ? {} : { description }),
  };
}

// Cross-field: a Project in deletion reports its deletion, and only then.
function requireProjectLifecycle(value: Project, status: number): Project {
  if ((value.lifecycle === "deleting") !== (value.deletion !== undefined)) {
    throw invalidProjectResponse(status);
  }
  return value;
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
    parsed.port === "0" ||
    parsed.search !== "" ||
    parsed.hash !== "" ||
    url.includes("?") ||
    url.includes("#")
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

// Cross-field: the ETag header names the revision of the body.
function requireETag(response: Response, revision: string): void {
  if (response.headers.get("ETag") !== `"${revision}"`) {
    throw invalidProjectResponse(response.status);
  }
}

export async function listProjects(
  api: PublicAPI,
  request: ProjectPageRequest = {},
  signal?: AbortSignal,
): Promise<ProjectPage> {
  const result = await api.request((client) =>
    client.GET("/v1/projects", {
      ...(signal === undefined ? {} : { signal }),
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
  for (const project of page.items) {
    requireProjectLifecycle(project, result.response.status);
  }
  return page;
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
  const project = requireProjectLifecycle(
    requireData(result),
    result.response.status,
  );
  // Cross-field: the Server answered for the requested Project.
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
  const project = requireProjectLifecycle(
    requireData(result),
    result.response.status,
  );
  // Cross-field: the created Project has the requested kind.
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
    ...(options.request.activeChecks === undefined
      ? {}
      : {
          activeChecks: requireActiveChecksPolicy(options.request.activeChecks),
        }),
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
  const project = requireProjectLifecycle(
    requireData(result),
    result.response.status,
  );
  // Cross-field: the Server answered for the requested Project.
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
  const project = requireProjectLifecycle(
    requireData(result),
    result.response.status,
  );
  // Cross-field: the requested Project is now in deletion.
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
  if (
    request.limit !== undefined &&
    (!Number.isInteger(request.limit) ||
      request.limit < 1 ||
      request.limit > 100)
  )
    throw new TypeError("Run page size is invalid");
  if (request.state !== undefined && !RUN_STATES.includes(request.state))
    throw new TypeError("Run state is invalid");
  if (
    request.lifecycle !== undefined &&
    !["active", "terminal"].includes(request.lifecycle)
  )
    throw new TypeError("Run lifecycle is invalid");
  const result = await api.request((client) =>
    client.GET("/v1/projects/{projectId}/runs", {
      params: {
        path: { projectId: request.projectId },
        query: {
          limit: request.limit ?? RUN_PAGE_SIZE,
          ...(request.state === undefined ? {} : { state: request.state }),
          ...(request.lifecycle === undefined
            ? {}
            : { lifecycle: request.lifecycle }),
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}
