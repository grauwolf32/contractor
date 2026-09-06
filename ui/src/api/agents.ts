import type { PublicAPI } from "./client";
import { publicAPIError } from "./error";
import type { components } from "./generated/public";
import { CONFIG_ID_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

export type AgentInstructions = components["schemas"]["AgentInstructions"];

export function agentPath(name: string, version: string): string {
  return `/catalog/agents/${encodeURIComponent(name)}/${encodeURIComponent(version)}`;
}

export async function getAgentInstructions(
  api: PublicAPI,
  name: string,
  version: string,
  signal?: AbortSignal,
): Promise<AgentInstructions> {
  if (!CONFIG_ID_PATTERN.test(name) || !CONFIG_VERSION_PATTERN.test(version)) {
    throw new TypeError("Agent version is invalid");
  }
  const result = await api.request((client) =>
    client.GET(
      "/v1/configurations/agent-templates/{name}/versions/{version}/instructions",
      {
        params: { path: { name, version } },
        ...(signal === undefined ? {} : { signal }),
      },
    ),
  );
  if (result.data === undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
  const value = result.data;
  if (
    value.template?.templateId !== name ||
    value.template.version !== version ||
    !/^sha256:[a-f0-9]{64}$/.test(value.template.digest) ||
    !/^sha256:[a-f0-9]{64}$/.test(value.instructions?.digest ?? "") ||
    typeof value.instructions?.ref !== "string" ||
    typeof value.instructions.text !== "string" ||
    value.instructions.text.length === 0
  ) {
    throw new Error("Server returned invalid agent instructions");
  }
  return value;
}
