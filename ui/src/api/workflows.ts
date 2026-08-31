import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import { safeConfigurationPage, safeCredentialPage } from "./safe-resources";

export const WORKFLOW_PAGE_SIZE = 50;
export const REFERENCE_PAGE_SIZE = 50;
export const CONFIG_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
export const CONFIG_VERSION_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
const RESOURCE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;

export type WorkflowPage = components["schemas"]["WorkflowPage"];
export type WorkflowResource = components["schemas"]["WorkflowResource"];
export type WorkflowSummary = components["schemas"]["WorkflowSummary"];
export type ConfigurationKind = components["schemas"]["ConfigurationKind"];
export type ConfigurationPage = components["schemas"]["ConfigurationPage"];
export type ConfigurationResource =
  components["schemas"]["ConfigurationResource"];
export type CredentialPage = components["schemas"]["CredentialPage"];
export type CredentialResource = components["schemas"]["CredentialResource"];
export type CreateRunRequest = components["schemas"]["CreateRunRequest"];
export type CreateRunResponse = components["schemas"]["CreateRunResponse"];

export interface PageRequest {
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

function requireWorkflowIdentity(name: string, version: string): void {
  if (!CONFIG_ID_PATTERN.test(name) || !CONFIG_VERSION_PATTERN.test(version)) {
    throw new TypeError("Workflow identity is invalid");
  }
}

export async function listWorkflows(
  api: PublicAPI,
  request: PageRequest = {},
): Promise<WorkflowPage> {
  const result = await api.request((client) =>
    client.GET("/v1/workflows", {
      params: {
        query: {
          limit: WORKFLOW_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return requireData(result);
}

export async function getWorkflow(
  api: PublicAPI,
  name: string,
  version: string,
): Promise<WorkflowResource> {
  requireWorkflowIdentity(name, version);
  const result = await api.request((client) =>
    client.GET("/v1/workflows/{name}/versions/{version}", {
      params: { path: { name, version } },
    }),
  );
  return requireData(result);
}

export async function listConfigurations(
  api: PublicAPI,
  kind: ConfigurationKind,
  request: PageRequest = {},
): Promise<ConfigurationPage> {
  const result = await api.request((client) =>
    client.GET("/v1/configurations/{kind}", {
      params: {
        path: { kind },
        query: {
          limit: REFERENCE_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return safeConfigurationPage(requireData(result));
}

export async function listCredentials(
  api: PublicAPI,
  request: PageRequest = {},
): Promise<CredentialPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/credentials", {
      params: {
        query: {
          limit: REFERENCE_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  return safeCredentialPage(requireData(result));
}

function invalidRunResponse(status: number): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message: "Server returned an invalid Run creation response",
  });
}

export async function createRun(
  api: PublicAPI,
  request: CreateRunRequest,
  idempotencyKey: string,
): Promise<CreateRunResponse> {
  const mutationHeaders = api.mutationHeaders({ idempotencyKey });
  const validatedKey = mutationHeaders.get("Idempotency-Key");
  if (validatedKey === null) {
    throw new TypeError("Run creation requires an idempotency key");
  }
  const result = await api.request((client) =>
    client.POST("/v1/runs", {
      params: { header: { "Idempotency-Key": validatedKey } },
      body: request,
    }),
  );
  const response = requireData(result);
  if (
    result.response.status !== 202 ||
    !RESOURCE_ID_PATTERN.test(response.runId) ||
    ![
      "initializing",
      "running",
      "cancelling",
      "succeeded",
      "failed",
      "cancelled",
    ].includes(response.state)
  ) {
    throw invalidRunResponse(result.response.status);
  }
  return response;
}
