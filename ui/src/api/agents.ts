import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";
import { CONFIG_NAME_PATTERN, CONFIG_VERSION_PATTERN } from "./workflows";

type AgentInstructions = components["schemas"]["AgentInstructions"];
type AgentTemplateWorkflowBindingPage =
  components["schemas"]["AgentTemplateWorkflowBindingPage"];

interface AgentUsagePageRequest {
  cursor?: string;
  signal?: AbortSignal;
}

export function agentPath(name: string, version: string): string {
  return `/catalog/agents/${encodeURIComponent(name)}/${encodeURIComponent(version)}`;
}

export async function getAgentInstructions(
  api: PublicAPI,
  name: string,
  version: string,
  signal?: AbortSignal,
): Promise<AgentInstructions> {
  if (
    !CONFIG_NAME_PATTERN.test(name) ||
    !CONFIG_VERSION_PATTERN.test(version)
  ) {
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
  const value = requireData(result);
  // The Server answers for the requested Agent version.
  if (
    value.template.templateId !== name ||
    value.template.version !== version
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned instructions of another Agent version",
    );
  }
  return value;
}

export async function listAgentTemplateWorkflowBindings(
  api: PublicAPI,
  name: string,
  version: string,
  request: AgentUsagePageRequest = {},
): Promise<AgentTemplateWorkflowBindingPage> {
  if (
    !CONFIG_NAME_PATTERN.test(name) ||
    !CONFIG_VERSION_PATTERN.test(version)
  ) {
    throw new TypeError("Agent version is invalid");
  }
  const result = await api.request((client) =>
    client.GET(
      "/v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings",
      {
        params: {
          path: { name, version },
          query: {
            limit: 50,
            ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
          },
        },
        ...(request.signal === undefined ? {} : { signal: request.signal }),
      },
    ),
  );
  return requireData(result);
}
