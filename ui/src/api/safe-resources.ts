import type { components } from "./generated/public";

type ConfigurationResource = components["schemas"]["ConfigurationResource"];
type ConfigurationPage = components["schemas"]["ConfigurationPage"];
type CredentialResource = components["schemas"]["CredentialResource"];
type CredentialPage = components["schemas"]["CredentialPage"];
type ExecutionSelection = components["schemas"]["ExecutionConfigSelection"];

const ARTIFACT_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;

function agentSkillRefs(
  values: components["schemas"]["AgentSkillRef"][] | undefined,
): components["schemas"]["AgentSkillRef"][] | undefined {
  if (values === undefined) {
    return undefined;
  }
  if (
    values.length > 32 ||
    values.some(
      (value, index) =>
        value.namespace !== "skills" ||
        !ARTIFACT_NAME.test(value.name) ||
        (index > 0 && values[index - 1]!.name >= value.name),
    )
  ) {
    throw new TypeError("AgentTemplate Skills response is invalid");
  }
  return values.map((value) => ({
    namespace: "skills",
    name: value.name,
  }));
}

function configurationRef(value: ConfigurationResource["ref"]) {
  return {
    kind: value.kind,
    name: value.name,
    version: value.version,
    digest: value.digest,
  };
}

function modelPolicyRef(value: components["schemas"]["ModelPolicyRef"]) {
  return {
    policyId: value.policyId,
    version: value.version,
    digest: value.digest,
  };
}

function gatewayRef(value: components["schemas"]["LLMGatewayConfigRef"]) {
  return {
    gatewayId: value.gatewayId,
    version: value.version,
    digest: value.digest,
  };
}

function modelPolicyBody(value: components["schemas"]["ModelPolicyBody"]) {
  return {
    model: value.model,
    ...(value.contextWindowTokens === undefined
      ? {}
      : { contextWindowTokens: value.contextWindowTokens }),
    ...(value.maxOutputTokens === undefined
      ? {}
      : { maxOutputTokens: value.maxOutputTokens }),
    ...(value.maxModelCalls === undefined
      ? {}
      : { maxModelCalls: value.maxModelCalls }),
    ...(value.maxToolCalls === undefined
      ? {}
      : { maxToolCalls: value.maxToolCalls }),
    ...(value.maxWorkerCalls === undefined
      ? {}
      : { maxWorkerCalls: value.maxWorkerCalls }),
    ...(value.maxTotalTokens === undefined
      ? {}
      : { maxTotalTokens: value.maxTotalTokens }),
    ...(value.temperature === undefined
      ? {}
      : { temperature: value.temperature }),
  };
}

function gatewayBody(value: components["schemas"]["LLMGatewayBody"]) {
  return {
    protocol: value.protocol,
    url: value.url,
    ...(value.credentialManager === undefined
      ? {}
      : {
          credentialManager: {
            implementation: value.credentialManager.implementation,
            managementUrl: value.credentialManager.managementUrl,
          },
        }),
  };
}

function executionSelection(value: ExecutionSelection): ExecutionSelection {
  return {
    ...(value.modelPolicy === undefined
      ? {}
      : { modelPolicy: modelPolicyRef(value.modelPolicy) }),
    ...(value.llmGateway === undefined
      ? {}
      : { llmGateway: gatewayRef(value.llmGateway) }),
    ...(value.credential === undefined
      ? {}
      : {
          credential:
            value.credential === null
              ? null
              : { credentialId: value.credential.credentialId },
        }),
  };
}

function referenceSummary(value: ConfigurationResource["body"]) {
  const summary = value as {
    description?: string;
    refs?: string[];
  };
  return {
    ...(summary.description === undefined
      ? {}
      : { description: summary.description }),
    ...(summary.refs === undefined ? {} : { refs: [...summary.refs] }),
  };
}

export function safeConfigurationResource(
  value: ConfigurationResource,
): ConfigurationResource {
  const ref = configurationRef(value.ref);
  switch (ref.kind) {
    case "model-policies":
      return {
        ref,
        source: value.source,
        body: modelPolicyBody(
          value.body as components["schemas"]["ModelPolicyBody"],
        ),
      };
    case "llm-gateways":
      return {
        ref,
        source: value.source,
        body: gatewayBody(
          value.body as components["schemas"]["LLMGatewayBody"],
        ),
      };
    case "agent-templates": {
      const body = value.body as components["schemas"]["AgentTemplateBody"];
      if (body.runtime === undefined) {
        return {
          ref,
          source: value.source,
          body: referenceSummary(value.body),
        };
      }
      const skills = agentSkillRefs(body.skills);
      return {
        ref,
        source: value.source,
        body: {
          description: body.description,
          runtime: body.runtime,
          instructions: {
            ref: body.instructions.ref,
            digest: body.instructions.digest,
          },
          modelPolicy: modelPolicyRef(body.modelPolicy),
          ...(body.summarizer === undefined
            ? {}
            : {
                summarizer: {
                  modelPolicy: modelPolicyRef(body.summarizer.modelPolicy),
                  contextWindowRatio: body.summarizer.contextWindowRatio,
                  ...(body.summarizer.cumulativeBudget === undefined
                    ? {}
                    : { cumulativeBudget: body.summarizer.cumulativeBudget }),
                },
              }),
          ...(skills === undefined ? {} : { skills }),
          toolsets: body.toolsets.map((selection) => ({
            ref: selection.ref,
            tools: [...selection.tools],
          })),
          sandboxProfile: body.sandboxProfile,
        },
      };
    }
    case "execution-configs": {
      const body = value.body as components["schemas"]["ExecutionConfigBody"];
      if ("description" in value.body || "refs" in value.body) {
        return {
          ref,
          source: value.source,
          body: referenceSummary(value.body),
        };
      }
      const agents = Object.fromEntries(
        Object.entries(body.agents ?? {}).map(([name, selection]) => [
          name,
          executionSelection(selection),
        ]),
      );
      return {
        ref,
        source: value.source,
        body: {
          ...(body.planner === undefined
            ? {}
            : { planner: executionSelection(body.planner) }),
          ...(Object.keys(agents).length === 0 ? {} : { agents }),
        },
      };
    }
  }
}

function pageInfo(value: components["schemas"]["PageInfo"]) {
  return {
    hasMore: value.hasMore,
    ...(value.nextCursor === undefined ? {} : { nextCursor: value.nextCursor }),
  };
}

export function safeConfigurationPage(
  value: ConfigurationPage,
): ConfigurationPage {
  return {
    items: value.items.map(safeConfigurationResource),
    page: pageInfo(value.page),
  };
}

export function safeCredentialResource(
  value: CredentialResource,
): CredentialResource {
  const policy = value.effectivePolicy;
  return {
    credentialId: value.credentialId,
    llmGateway: gatewayRef(value.llmGateway),
    ...(value.label === undefined ? {} : { label: value.label }),
    createdAt: value.createdAt,
    effectivePolicy: {
      modelPolicies: policy.modelPolicies.map(modelPolicyRef),
      models: [...policy.models],
      ...(policy.maxBudget === undefined
        ? {}
        : { maxBudget: policy.maxBudget }),
      ...(policy.budgetDuration === undefined
        ? {}
        : { budgetDuration: policy.budgetDuration }),
      ...(policy.tpmLimit === undefined ? {} : { tpmLimit: policy.tpmLimit }),
      ...(policy.rpmLimit === undefined ? {} : { rpmLimit: policy.rpmLimit }),
      ...(policy.maxParallelRequests === undefined
        ? {}
        : { maxParallelRequests: policy.maxParallelRequests }),
    },
    ...(value.consumption === undefined
      ? {}
      : {
          consumption: {
            ...(value.consumption.spend === undefined
              ? {}
              : { spend: value.consumption.spend }),
            ...(value.consumption.requests === undefined
              ? {}
              : { requests: value.consumption.requests }),
            ...(value.consumption.inputTokens === undefined
              ? {}
              : { inputTokens: value.consumption.inputTokens }),
            ...(value.consumption.outputTokens === undefined
              ? {}
              : { outputTokens: value.consumption.outputTokens }),
            ...(value.consumption.observedAt === undefined
              ? {}
              : { observedAt: value.consumption.observedAt }),
          },
        }),
  };
}

export function safeCredentialPage(value: CredentialPage): CredentialPage {
  return {
    items: value.items.map(safeCredentialResource),
    page: pageInfo(value.page),
  };
}
