import type { PublicAPI } from "./client";
import { schemaViolation } from "./contract";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

export type ModelSelectionValue = components["schemas"]["ModelSelectionValue"];
type ModelSelectionInfo = components["schemas"]["ModelSelectionInfo"];
interface ModelSelectionSnapshot {
  resource: ModelSelectionInfo;
  etag: string | undefined;
}
export const MODEL_ROLES = [
  "planner",
  "worker",
  "summarizer",
  "audit_completion_worker",
] as const;

export function sameModelSelection(
  a: ModelSelectionValue,
  b: ModelSelectionValue,
): boolean {
  return (
    a.defaultModel === b.defaultModel &&
    MODEL_ROLES.every((role) => a.roles[role] === b.roles[role])
  );
}

export function compatibleModelSelection(
  value: ModelSelectionValue,
  models: ModelSelectionInfo["models"],
): boolean {
  return (
    models.some((model) => model.id === value.defaultModel) &&
    MODEL_ROLES.every((role) =>
      models.some(
        (model) =>
          model.id === (value.roles[role] ?? value.defaultModel) &&
          model.policies[role] !== undefined,
      ),
    )
  );
}

function requireModelSelection(
  value: ModelSelectionInfo,
  response: Response,
): ModelSelectionSnapshot {
  const etag = response.headers.get("ETag") ?? undefined;
  if (
    schemaViolation("ModelSelectionInfo", value) !== undefined ||
    (value.configured &&
      (value.selection === undefined ||
        value.revision === undefined ||
        value.catalogDigest === undefined ||
        value.updatedAt === undefined ||
        value.models.length === 0 ||
        etag !== `"${value.revision}:${value.catalogDigest.slice(7)}"` ||
        !compatibleModelSelection(value.selection, value.models))) ||
    (!value.configured && value.models.length !== 0)
  ) {
    throw invalidAPIResponse(
      response.status,
      "Server returned invalid model settings",
    );
  }
  return { resource: value, etag };
}

export async function getModelSelection(
  api: PublicAPI,
): Promise<ModelSelectionSnapshot> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/settings/models"),
  );
  return requireModelSelection(requireData(result), result.response);
}

export async function replaceModelSelection(
  api: PublicAPI,
  value: ModelSelectionValue,
  etag: string,
): Promise<ModelSelectionSnapshot> {
  const result = await api.request((client) =>
    client.PUT("/v1/operations/settings/models", {
      params: { header: { "If-Match": etag } },
      body: value,
    }),
  );
  const saved = requireModelSelection(requireData(result), result.response);
  if (
    saved.resource.selection === undefined ||
    !sameModelSelection(saved.resource.selection, value)
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned a different model selection after replacement",
    );
  }
  return saved;
}
