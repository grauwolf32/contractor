import type { PublicAPI } from "./client";
import { PublicAPIError, publicAPIError } from "./error";
import type { components } from "./generated/public";
import {
  safeConfigurationResource,
  safeCredentialResource,
} from "./safe-resources";
import {
  CONFIG_ID_PATTERN,
  CONFIG_VERSION_PATTERN,
  listConfigurations,
  listCredentials,
} from "./workflows";

export const OPERATIONS_PAGE_SIZE = 50;
export const CONFIGURATION_KINDS = [
  "agent-templates",
  "execution-configs",
  "model-policies",
  "llm-gateways",
] as const satisfies readonly ConfigurationKind[];
export const WRITABLE_CONFIGURATION_KINDS = [
  "model-policies",
  "llm-gateways",
] as const satisfies readonly WritableConfigurationKind[];
export const MANAGED_CONFIG_NAME_PATTERN = CONFIG_ID_PATTERN;
export const BUDGET_DURATION_PATTERN = /^[1-9][0-9]*(?:s|m|h|d|mo)$/;

const RESOURCE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
const UNSIGNED_DECIMAL = /^(?:0|[1-9][0-9]*)$/;
const RUNTIME_CAPABILITY_REF_PATTERN =
  /^[a-z][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9._+-]*$/;
const MAX_RUNTIME_CAPABILITY_REFS = 128;
const MAX_RUNTIME_TOOLSETS = 128;
const MAX_RUNTIME_TOOLS_PER_TOOLSET = 256;

export type OperationsSnapshot = components["schemas"]["OperationsSnapshot"];
export type OperationsCursor = components["schemas"]["SnapshotCursor"];
export type RuntimeAgentObservation =
  components["schemas"]["RuntimeAgentObservation"];
export type AllocationObservation =
  components["schemas"]["AllocationObservation"];
export type RuntimeAgentPage = components["schemas"]["RuntimeAgentPage"];
export type AllocationPage = components["schemas"]["AllocationPage"];
export type ConfigurationKind = components["schemas"]["ConfigurationKind"];
export type ConfigurationResource =
  components["schemas"]["ConfigurationResource"];
export type ConfigurationPage = components["schemas"]["ConfigurationPage"];
export type PublishConfigurationRequest =
  components["schemas"]["PublishConfigurationRequest"];
export type ModelPolicyBody = components["schemas"]["ModelPolicyBody"];
export type LLMGatewayBody = components["schemas"]["LLMGatewayBody"];
export type AgentTemplateBody = components["schemas"]["AgentTemplateBody"];
export type ExecutionConfigBody = components["schemas"]["ExecutionConfigBody"];
export type ModelPolicyRef = components["schemas"]["ModelPolicyRef"];
export type LLMGatewayConfigRef = components["schemas"]["LLMGatewayConfigRef"];
export type CredentialResource = components["schemas"]["CredentialResource"];
export type CredentialPage = components["schemas"]["CredentialPage"];
export type CreateCredentialRequest =
  components["schemas"]["CreateCredentialRequest"];
export type RuntimeConfigDocument =
  components["schemas"]["RuntimeConfigDocument"];
export type RuntimeConfigResource =
  components["schemas"]["RuntimeConfigResource"];
export type RuntimeConfigPage = components["schemas"]["RuntimeConfigPage"];
export type RuntimeLabelBinding = components["schemas"]["RuntimeLabelBinding"];
export type RuntimeLabelPage = components["schemas"]["RuntimeLabelPage"];
export type RuntimeCredentialMetadata =
  components["schemas"]["RuntimeCredentialMetadata"];
export type RuntimeCredentialPage =
  components["schemas"]["RuntimeCredentialPage"];
export type CreateRuntimeCredentialRequest =
  components["schemas"]["CreateRuntimeCredentialRequest"];

export type WritableConfigurationKind = "model-policies" | "llm-gateways";

export interface CursorPageRequest {
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

function invalidResponse(status: number, message: string): PublicAPIError {
  return new PublicAPIError({
    status,
    code: "invalid_api_response",
    message,
  });
}

function requireResourceID(label: string, value: string): void {
  if (!RESOURCE_ID_PATTERN.test(value)) {
    throw new TypeError(`${label} is invalid`);
  }
}

function requireConfigIdentity(name: string, version?: string): void {
  if (
    !CONFIG_ID_PATTERN.test(name) ||
    (version !== undefined && !CONFIG_VERSION_PATTERN.test(version))
  ) {
    throw new TypeError("Configuration identity is invalid");
  }
}

function requirePublishedRef(ref: {
  version: string;
  digest: string;
  policyId?: string;
  gatewayId?: string;
}): void {
  const name = ref.policyId ?? ref.gatewayId ?? "";
  requireConfigIdentity(name, ref.version);
  if (!DIGEST_PATTERN.test(ref.digest)) {
    throw new TypeError("Published configuration digest is invalid");
  }
}

function requireRuntimeIdentity(name: string, version?: string): void {
  if (
    !/^[a-z][a-z0-9_-]{0,62}$/.test(name) ||
    (version !== undefined &&
      !/^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/.test(version))
  ) {
    throw new TypeError("Runtime configuration identity is invalid");
  }
}

function requireRuntimeCredentialId(value: string): void {
  if (!/^[a-z][a-z0-9_-]{0,127}$/.test(value)) {
    throw new TypeError("Runtime credential ID is invalid");
  }
}

function safeRuntimeRef(
  ref: components["schemas"]["RuntimeConfigRef"],
): components["schemas"]["RuntimeConfigRef"] {
  requireRuntimeIdentity(ref.name, ref.version);
  if (!DIGEST_PATTERN.test(ref.digest)) {
    throw new TypeError("RuntimeConfig ref is invalid");
  }
  return { name: ref.name, version: ref.version, digest: ref.digest };
}

function safeRuntimeTelemetry(
  value: components["schemas"]["RuntimeTelemetryConfig"],
): components["schemas"]["RuntimeTelemetryConfig"] {
  const endpoint = new URL(value.endpoint);
  if (
    !["http:", "https:"].includes(endpoint.protocol) ||
    endpoint.username !== "" ||
    endpoint.password !== "" ||
    endpoint.search !== "" ||
    endpoint.hash !== ""
  ) {
    throw new TypeError("Runtime telemetry endpoint is invalid");
  }
  return {
    adapter: value.adapter,
    endpoint: endpoint.toString(),
    ...(value.credential === undefined ? {} : { credential: value.credential }),
    ...(value.captureContent === undefined
      ? {}
      : { captureContent: value.captureContent }),
    ...(value.flushTimeoutSeconds === undefined
      ? {}
      : { flushTimeoutSeconds: value.flushTimeoutSeconds }),
  };
}

function safeRuntimeDocument(
  value: RuntimeConfigDocument,
): RuntimeConfigDocument {
  requireRuntimeIdentity(value.metadata.name, value.metadata.version);
  const worker = value.spec.worker;
  const planner = value.spec.planner;
  return {
    apiVersion: value.apiVersion,
    kind: value.kind,
    metadata: {
      name: value.metadata.name,
      version: value.metadata.version,
    },
    spec: {
      ...(worker === undefined
        ? {}
        : {
            worker: {
              ...(worker.llmGateway === undefined
                ? {}
                : {
                    llmGateway:
                      worker.llmGateway === null
                        ? null
                        : {
                            ...(worker.llmGateway.gateway === undefined
                              ? {}
                              : {
                                  gateway:
                                    typeof worker.llmGateway.gateway ===
                                    "string"
                                      ? worker.llmGateway.gateway
                                      : worker.llmGateway.gateway === null
                                        ? null
                                        : {
                                            gatewayId:
                                              worker.llmGateway.gateway
                                                .gatewayId,
                                            version:
                                              worker.llmGateway.gateway.version,
                                            digest:
                                              worker.llmGateway.gateway.digest,
                                          },
                                }),
                            ...(worker.llmGateway.credential === undefined
                              ? {}
                              : {
                                  credential: worker.llmGateway.credential,
                                }),
                          },
                  }),
              ...(worker.telemetry === undefined
                ? {}
                : {
                    telemetry:
                      worker.telemetry === null
                        ? null
                        : safeRuntimeTelemetry(worker.telemetry),
                  }),
              ...(worker.httpProxy === undefined
                ? {}
                : {
                    httpProxy:
                      worker.httpProxy === null
                        ? null
                        : {
                            adapter: worker.httpProxy.adapter,
                            proxyUrl: worker.httpProxy.proxyUrl,
                            ...(worker.httpProxy.credential === undefined
                              ? {}
                              : { credential: worker.httpProxy.credential }),
                            ...(worker.httpProxy.caBundlePem === undefined
                              ? {}
                              : { caBundlePem: worker.httpProxy.caBundlePem }),
                            targets: [...worker.httpProxy.targets],
                          },
                  }),
            },
          }),
      ...(planner === undefined
        ? {}
        : {
            planner: {
              ...(planner.telemetry === undefined
                ? {}
                : {
                    telemetry:
                      planner.telemetry === null
                        ? null
                        : safeRuntimeTelemetry(planner.telemetry),
                  }),
            },
          }),
    },
  };
}

function safeRuntimeConfigResource(
  value: RuntimeConfigResource,
): RuntimeConfigResource {
  const ref = safeRuntimeRef(value.ref);
  const document = safeRuntimeDocument(value.document);
  if (
    ref.name !== document.metadata.name ||
    ref.version !== document.metadata.version
  ) {
    throw new TypeError("RuntimeConfig resource identity is invalid");
  }
  return {
    ref,
    document,
    builtIn: value.builtIn,
    createdBy: value.createdBy,
    createdAt: value.createdAt,
  };
}

function safeRuntimeLabel(value: RuntimeLabelBinding): RuntimeLabelBinding {
  requireRuntimeIdentity(value.label);
  if (!/^[1-9][0-9]{0,19}$/.test(value.revision)) {
    throw new TypeError("Runtime label revision is invalid");
  }
  return {
    label: value.label,
    config: safeRuntimeRef(value.config),
    revision: value.revision,
    createdBy: value.createdBy,
    createdAt: value.createdAt,
    updatedBy: value.updatedBy,
    updatedAt: value.updatedAt,
  };
}

function safeRuntimeCredential(
  value: RuntimeCredentialMetadata,
): RuntimeCredentialMetadata {
  requireRuntimeCredentialId(value.credentialId);
  if (
    !["otlp-headers@1", "http-proxy-basic@1", "http-proxy-bearer@1"].includes(
      value.kind,
    )
  ) {
    throw new TypeError("Runtime credential kind is invalid");
  }
  return {
    credentialId: value.credentialId,
    kind: value.kind,
    createdBy: value.createdBy,
    createdAt: value.createdAt,
  };
}

function idempotencyHeader(api: PublicAPI, key: string): string {
  const result = api
    .mutationHeaders({ idempotencyKey: key })
    .get("Idempotency-Key");
  if (result === null) {
    throw new TypeError("Mutation requires an idempotency key");
  }
  return result;
}

function validOperationsCursor(cursor: OperationsCursor): boolean {
  return (
    RESOURCE_ID_PATTERN.test(cursor.generation) &&
    cursor.revision.length <= 20 &&
    UNSIGNED_DECIMAL.test(cursor.revision)
  );
}

function safeReason(value: { code: string; retryable: boolean } | undefined) {
  return value === undefined
    ? undefined
    : { code: value.code, retryable: value.retryable };
}

function safeRuntimeAgent(
  value: RuntimeAgentObservation,
): RuntimeAgentObservation {
  const reason = safeReason(value.reconciliationReason);
  const runtimes = safeCapabilityRefs(value.supportedRuntimes, true);
  const sandboxes = safeCapabilityRefs(value.supportedSandboxProfiles, true);
  if (
    !Array.isArray(value.supportedToolsets) ||
    value.supportedToolsets.length > MAX_RUNTIME_TOOLSETS
  ) {
    throw new TypeError("Runtime Agent capabilities are invalid");
  }
  const seenToolsets = new Set<string>();
  const toolsets = value.supportedToolsets.map((capability) => {
    if (
      capability === null ||
      typeof capability !== "object" ||
      typeof capability.ref !== "string" ||
      capability.ref.length > 256 ||
      !RUNTIME_CAPABILITY_REF_PATTERN.test(capability.ref) ||
      seenToolsets.has(capability.ref) ||
      !Array.isArray(capability.tools) ||
      capability.tools.length === 0 ||
      capability.tools.length > MAX_RUNTIME_TOOLS_PER_TOOLSET
    ) {
      throw new TypeError("Runtime Agent capabilities are invalid");
    }
    seenToolsets.add(capability.ref);
    const tools = capability.tools.map((tool) => {
      if (typeof tool !== "string" || !CONFIG_ID_PATTERN.test(tool)) {
        throw new TypeError("Runtime Agent capabilities are invalid");
      }
      return tool;
    });
    if (new Set(tools).size !== tools.length) {
      throw new TypeError("Runtime Agent capabilities are invalid");
    }
    return { ref: capability.ref, tools: tools.sort() };
  });
  toolsets.sort((left, right) => left.ref.localeCompare(right.ref));
  return {
    instanceId: value.instanceId,
    softwareVersion: value.softwareVersion,
    supportedRuntimes: runtimes,
    supportedToolsets: toolsets,
    supportedSandboxProfiles: sandboxes,
    observedState: value.observedState,
    slotState: value.slotState,
    ...(value.lastAcceptedHeartbeat === undefined
      ? {}
      : { lastAcceptedHeartbeat: value.lastAcceptedHeartbeat }),
    ...(value.confirmedLeaseUntil === undefined
      ? {}
      : { confirmedLeaseUntil: value.confirmedLeaseUntil }),
    ...(value.currentAllocationId === undefined
      ? {}
      : { currentAllocationId: value.currentAllocationId }),
    ...(value.authoritativeAllocationId === undefined
      ? {}
      : { authoritativeAllocationId: value.authoritativeAllocationId }),
    ...(reason === undefined ? {} : { reconciliationReason: reason }),
  };
}

function safeCapabilityRefs(values: string[], required: boolean): string[] {
  if (
    !Array.isArray(values) ||
    (required && values.length === 0) ||
    values.length > MAX_RUNTIME_CAPABILITY_REFS
  ) {
    throw new TypeError("Runtime Agent capabilities are invalid");
  }
  const result = values.map((value) => {
    if (
      typeof value !== "string" ||
      value.length > 256 ||
      !RUNTIME_CAPABILITY_REF_PATTERN.test(value)
    ) {
      throw new TypeError("Runtime Agent capabilities are invalid");
    }
    return value;
  });
  if (new Set(result).size !== result.length) {
    throw new TypeError("Runtime Agent capabilities are invalid");
  }
  return result.sort();
}

function safeAllocation(value: AllocationObservation): AllocationObservation {
  const reason = safeReason(value.reason);
  const origins = value.executionConfig.origins;
  const llmGateway = value.executionConfig.llmGateway;
  if (llmGateway === undefined) {
    throw new TypeError("active Allocation has no resolved LLM Gateway");
  }
  return {
    allocationId: value.allocationId,
    runId: value.runId,
    stageExecutionId: value.stageExecutionId,
    runtimeAgentInstanceId: value.runtimeAgentInstanceId,
    logicalWorker: value.logicalWorker,
    agentTemplate: {
      templateId: value.agentTemplate.templateId,
      version: value.agentTemplate.version,
      digest: value.agentTemplate.digest,
    },
    executionConfig: {
      modelPolicy: {
        policyId: value.executionConfig.modelPolicy.policyId,
        version: value.executionConfig.modelPolicy.version,
        digest: value.executionConfig.modelPolicy.digest,
      },
      llmGateway: {
        gatewayId: llmGateway.gatewayId,
        version: llmGateway.version,
        digest: llmGateway.digest,
      },
      ...(value.executionConfig.credential === undefined
        ? {}
        : {
            credential: {
              credentialId: value.executionConfig.credential.credentialId,
            },
          }),
      ...(origins === undefined
        ? {}
        : {
            origins: {
              modelPolicy: origins.modelPolicy,
              ...(origins.llmGateway === undefined
                ? {}
                : { llmGateway: origins.llmGateway }),
              ...(origins.credential === undefined
                ? {}
                : { credential: origins.credential }),
            },
          }),
    },
    authoritativePhase: value.authoritativePhase,
    observedPhase: value.observedPhase,
    ...(reason === undefined ? {} : { reason }),
    metrics: {
      reportsComplete: value.metrics.reportsComplete,
      modelCalls: value.metrics.modelCalls,
      inputTokens: value.metrics.inputTokens,
      outputTokens: value.metrics.outputTokens,
      totalTokens: value.metrics.totalTokens,
      toolCalls: value.metrics.toolCalls,
      toolFailures: value.metrics.toolFailures,
      errorCount: value.metrics.errorCount,
      truncated: value.metrics.truncated,
    },
    ...(value.exhaustedDimension === undefined
      ? {}
      : { exhaustedDimension: value.exhaustedDimension }),
  };
}

function safeCursor(cursor: OperationsCursor): OperationsCursor {
  return { generation: cursor.generation, revision: cursor.revision };
}

function safePageInfo(page: { hasMore: boolean; nextCursor?: string }) {
  return {
    hasMore: page.hasMore,
    ...(page.nextCursor === undefined ? {} : { nextCursor: page.nextCursor }),
  };
}

export { listConfigurations, listCredentials };

export async function getOperationsSnapshot(
  api: PublicAPI,
): Promise<OperationsSnapshot> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/snapshot"),
  );
  const snapshot = requireData(result);
  if (!validOperationsCursor(snapshot.cursor)) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid Operations snapshot",
    );
  }
  return {
    cursor: safeCursor(snapshot.cursor),
    runtimeAgents: snapshot.runtimeAgents.map(safeRuntimeAgent),
    allocations: snapshot.allocations.map(safeAllocation),
  };
}

export async function listRuntimeAgents(
  api: PublicAPI,
  request: CursorPageRequest = {},
): Promise<RuntimeAgentPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-agents", {
      params: {
        query: {
          limit: OPERATIONS_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  if (!validOperationsCursor(page.cursor)) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid Runtime Agent page",
    );
  }
  return {
    cursor: safeCursor(page.cursor),
    items: page.items.map(safeRuntimeAgent),
    page: safePageInfo(page.page),
  };
}

export async function listAllocations(
  api: PublicAPI,
  request: CursorPageRequest = {},
): Promise<AllocationPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/allocations", {
      params: {
        query: {
          limit: OPERATIONS_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  if (!validOperationsCursor(page.cursor)) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid allocation page",
    );
  }
  return {
    cursor: safeCursor(page.cursor),
    items: page.items.map(safeAllocation),
    page: safePageInfo(page.page),
  };
}

export async function getConfiguration(
  api: PublicAPI,
  kind: ConfigurationKind,
  name: string,
  version: string,
): Promise<ConfigurationResource> {
  requireConfigIdentity(name, version);
  const result = await api.request((client) =>
    client.GET("/v1/configurations/{kind}/{name}/versions/{version}", {
      params: { path: { kind, name, version } },
    }),
  );
  const resource = safeConfigurationResource(requireData(result));
  if (
    resource.ref.kind !== kind ||
    resource.ref.name !== name ||
    resource.ref.version !== version ||
    !DIGEST_PATTERN.test(resource.ref.digest)
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid configuration resource",
    );
  }
  return resource;
}

export async function publishConfiguration(
  api: PublicAPI,
  kind: WritableConfigurationKind,
  request: PublishConfigurationRequest,
  idempotencyKey: string,
): Promise<ConfigurationResource> {
  if (!WRITABLE_CONFIGURATION_KINDS.includes(kind)) {
    throw new TypeError("Configuration kind is read-only");
  }
  if (
    !MANAGED_CONFIG_NAME_PATTERN.test(request.name) ||
    !CONFIG_VERSION_PATTERN.test(request.version)
  ) {
    throw new TypeError("Managed configuration identity is invalid");
  }
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.POST("/v1/configurations/{kind}", {
      params: {
        path: { kind },
        header: { "Idempotency-Key": header },
      },
      body: request,
    }),
  );
  const resource = safeConfigurationResource(requireData(result));
  if (
    result.response.status !== 201 ||
    resource.ref.kind !== kind ||
    resource.ref.name !== request.name ||
    resource.ref.version !== request.version ||
    !DIGEST_PATTERN.test(resource.ref.digest)
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid configuration publication response",
    );
  }
  return resource;
}

export async function getCredential(
  api: PublicAPI,
  credentialId: string,
): Promise<CredentialResource> {
  requireConfigIdentity(credentialId);
  const result = await api.request((client) =>
    client.GET("/v1/operations/credentials/{credentialId}", {
      params: { path: { credentialId } },
    }),
  );
  const credential = safeCredentialResource(requireData(result));
  if (credential.credentialId !== credentialId) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid credential resource",
    );
  }
  requirePublishedRef(credential.llmGateway);
  credential.effectivePolicy.modelPolicies.forEach(requirePublishedRef);
  return credential;
}

export async function createCredential(
  api: PublicAPI,
  request: CreateCredentialRequest,
  idempotencyKey: string,
): Promise<CredentialResource> {
  requireConfigIdentity(request.credentialId);
  requirePublishedRef(request.llmGateway);
  request.gatewayPolicy.modelPolicies.forEach(requirePublishedRef);
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.POST("/v1/operations/credentials", {
      params: { header: { "Idempotency-Key": header } },
      body: request,
    }),
  );
  const credential = safeCredentialResource(requireData(result));
  if (
    result.response.status !== 201 ||
    credential.credentialId !== request.credentialId ||
    credential.llmGateway.gatewayId !== request.llmGateway.gatewayId ||
    credential.llmGateway.version !== request.llmGateway.version ||
    credential.llmGateway.digest !== request.llmGateway.digest
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid credential creation response",
    );
  }
  credential.effectivePolicy.modelPolicies.forEach(requirePublishedRef);
  return credential;
}

export async function deleteCredential(
  api: PublicAPI,
  credentialId: string,
  idempotencyKey: string,
): Promise<void> {
  requireConfigIdentity(credentialId);
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.DELETE("/v1/operations/credentials/{credentialId}", {
      params: {
        path: { credentialId },
        header: { "Idempotency-Key": header },
      },
    }),
  );
  if (result.response.status !== 204 || result.error !== undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
}

export async function listRuntimeConfigs(
  api: PublicAPI,
  request: CursorPageRequest = {},
): Promise<RuntimeConfigPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-configs", {
      params: {
        query: {
          limit: OPERATIONS_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  return {
    items: page.items.map(safeRuntimeConfigResource),
    page: safePageInfo(page.page),
  };
}

export async function getRuntimeConfig(
  api: PublicAPI,
  name: string,
  version: string,
): Promise<RuntimeConfigResource> {
  requireRuntimeIdentity(name, version);
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-configs/{name}/versions/{version}", {
      params: { path: { name, version } },
    }),
  );
  const resource = safeRuntimeConfigResource(requireData(result));
  if (resource.ref.name !== name || resource.ref.version !== version) {
    throw invalidResponse(
      result.response.status,
      "Server returned the wrong RuntimeConfig resource",
    );
  }
  return resource;
}

export async function publishRuntimeConfig(
  api: PublicAPI,
  document: RuntimeConfigDocument,
  idempotencyKey: string,
): Promise<RuntimeConfigResource> {
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.POST("/v1/operations/runtime-configs", {
      params: { header: { "Idempotency-Key": header } },
      body: document,
    }),
  );
  const resource = safeRuntimeConfigResource(requireData(result));
  if (
    result.response.status !== 201 ||
    resource.ref.name !== document.metadata.name ||
    resource.ref.version !== document.metadata.version
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid RuntimeConfig publication response",
    );
  }
  return resource;
}

export async function listRuntimeLabels(
  api: PublicAPI,
  request: CursorPageRequest = {},
): Promise<RuntimeLabelPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-labels", {
      params: {
        query: {
          limit: OPERATIONS_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  return {
    items: page.items.map(safeRuntimeLabel),
    page: safePageInfo(page.page),
  };
}

export async function getRuntimeLabel(
  api: PublicAPI,
  label: string,
): Promise<RuntimeLabelBinding> {
  requireRuntimeIdentity(label);
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-labels/{label}", {
      params: { path: { label } },
    }),
  );
  const binding = safeRuntimeLabel(requireData(result));
  if (
    binding.label !== label ||
    result.response.headers.get("etag") !== `"${binding.revision}"`
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid Runtime label response",
    );
  }
  return binding;
}

export async function putRuntimeLabel(
  api: PublicAPI,
  label: string,
  config: components["schemas"]["RuntimeConfigRef"],
  idempotencyKey: string,
  expectedRevision?: string,
): Promise<RuntimeLabelBinding> {
  requireRuntimeIdentity(label);
  const safeConfig = safeRuntimeRef(config);
  if (
    expectedRevision !== undefined &&
    !/^[1-9][0-9]{0,19}$/.test(expectedRevision)
  ) {
    throw new TypeError("Runtime label revision is invalid");
  }
  const headers = api.mutationHeaders({
    idempotencyKey,
    ...(expectedRevision === undefined
      ? { ifNoneMatch: "*" }
      : { ifMatch: `"${expectedRevision}"` }),
  });
  const key = headers.get("Idempotency-Key");
  if (key === null) {
    throw new TypeError("Mutation requires an idempotency key");
  }
  const result = await api.request((client) =>
    client.PUT("/v1/operations/runtime-labels/{label}", {
      params: {
        path: { label },
        header: {
          "Idempotency-Key": key,
          ...(expectedRevision === undefined
            ? { "If-None-Match": "*" as const }
            : { "If-Match": `"${expectedRevision}"` }),
        },
      },
      body: { config: safeConfig },
    }),
  );
  const binding = safeRuntimeLabel(requireData(result));
  if (
    ![200, 201].includes(result.response.status) ||
    binding.label !== label ||
    result.response.headers.get("etag") !== `"${binding.revision}"`
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid Runtime label mutation response",
    );
  }
  return binding;
}

export async function deleteRuntimeLabel(
  api: PublicAPI,
  label: string,
  revision: string,
  idempotencyKey: string,
): Promise<void> {
  requireRuntimeIdentity(label);
  if (!/^[1-9][0-9]{0,19}$/.test(revision)) {
    throw new TypeError("Runtime label revision is invalid");
  }
  const headers = api.mutationHeaders({
    idempotencyKey,
    ifMatch: `"${revision}"`,
  });
  const key = headers.get("Idempotency-Key");
  if (key === null) {
    throw new TypeError("Mutation requires an idempotency key");
  }
  const result = await api.request((client) =>
    client.DELETE("/v1/operations/runtime-labels/{label}", {
      params: {
        path: { label },
        header: { "Idempotency-Key": key, "If-Match": `"${revision}"` },
      },
    }),
  );
  if (result.response.status !== 204 || result.error !== undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
}

export async function listRuntimeCredentials(
  api: PublicAPI,
  request: CursorPageRequest = {},
): Promise<RuntimeCredentialPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-credentials", {
      params: {
        query: {
          limit: OPERATIONS_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  return {
    items: page.items.map(safeRuntimeCredential),
    page: safePageInfo(page.page),
  };
}

export async function getRuntimeCredential(
  api: PublicAPI,
  credentialId: string,
): Promise<RuntimeCredentialMetadata> {
  requireRuntimeCredentialId(credentialId);
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-credentials/{credentialId}", {
      params: { path: { credentialId } },
    }),
  );
  const metadata = safeRuntimeCredential(requireData(result));
  if (metadata.credentialId !== credentialId) {
    throw invalidResponse(
      result.response.status,
      "Server returned the wrong Runtime credential metadata",
    );
  }
  return metadata;
}

export async function createRuntimeCredential(
  api: PublicAPI,
  request: CreateRuntimeCredentialRequest,
  idempotencyKey: string,
): Promise<RuntimeCredentialMetadata> {
  requireRuntimeCredentialId(request.credentialId);
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.POST("/v1/operations/runtime-credentials", {
      params: { header: { "Idempotency-Key": header } },
      body: request,
    }),
  );
  const metadata = safeRuntimeCredential(requireData(result));
  if (
    result.response.status !== 201 ||
    metadata.credentialId !== request.credentialId ||
    metadata.kind !== request.kind
  ) {
    throw invalidResponse(
      result.response.status,
      "Server returned an invalid Runtime credential creation response",
    );
  }
  return metadata;
}

export async function deleteRuntimeCredential(
  api: PublicAPI,
  credentialId: string,
  idempotencyKey: string,
): Promise<void> {
  requireRuntimeCredentialId(credentialId);
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.DELETE("/v1/operations/runtime-credentials/{credentialId}", {
      params: {
        path: { credentialId },
        header: { "Idempotency-Key": header },
      },
    }),
  );
  if (result.response.status !== 204 || result.error !== undefined) {
    throw publicAPIError(result.response.status, result.error);
  }
}

export function isConfigurationKind(value: string): value is ConfigurationKind {
  return CONFIGURATION_KINDS.some((kind) => kind === value);
}

export function isWritableConfigurationKind(
  value: ConfigurationKind,
): value is WritableConfigurationKind {
  return WRITABLE_CONFIGURATION_KINDS.some((kind) => kind === value);
}

export function requireOperationsResourceID(
  label: string,
  value: string,
): void {
  requireResourceID(label, value);
}
