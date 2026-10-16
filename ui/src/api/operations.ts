import type { PublicAPI } from "./client";
import {
  CONFIG_ID,
  DIGEST,
  RUNTIME_AGENT_ID,
  RUNTIME_CONFIG_VERSION,
  RUNTIME_CREDENTIAL_ID,
  RUNTIME_INFRASTRUCTURE_ID,
  schemaEnum,
  stringRule,
} from "./contract";
import { invalidAPIResponse, publicAPIError, requireData } from "./error";
import { loadCompletePages } from "./pages";
import type { components } from "./generated/public";
import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
  listConfigurations,
  listCredentials,
  requireCanonicalConfiguration,
} from "./workflows";

const OPERATIONS_PAGE_SIZE = 50;
const CONFIGURATION_KINDS = schemaEnum<ConfigurationKind>("ConfigurationKind");
const WRITABLE_CONFIGURATION_KINDS = [
  "model-policies",
  "llm-gateways",
] as const satisfies readonly WritableConfigurationKind[];
/** LLM credential IDs are ConfigId (CreateCredentialRequest.credentialId). */
export const CREDENTIAL_ID = CONFIG_ID;
export const BUDGET_DURATION_PATTERN = stringRule(
  "GatewayPolicy",
  "properties",
  "budgetDuration",
);
// Telemetry export retry defaults the Runtime applies to an omitted bound.
const DEFAULT_INITIAL_BACKOFF_MILLISECONDS = 100;
const DEFAULT_MAX_BACKOFF_MILLISECONDS = 1000;

export type OperationsSnapshot = components["schemas"]["OperationsSnapshot"];
export type RuntimeAgentObservation =
  components["schemas"]["RuntimeAgentObservation"];
type WorkspaceCapabilities = components["schemas"]["WorkspaceCapabilities"];
export type AllocationObservation =
  components["schemas"]["AllocationObservation"];
export type ConfigurationKind = components["schemas"]["ConfigurationKind"];
export type ConfigurationResource =
  components["schemas"]["ConfigurationResource"];
export type PublishConfigurationRequest =
  components["schemas"]["PublishConfigurationRequest"];
export type ModelPolicyBody = components["schemas"]["ModelPolicyBody"];
export type LLMGatewayBody = components["schemas"]["LLMGatewayBody"];
export type AgentTemplateBody = components["schemas"]["AgentTemplateBody"];
export type ExecutionConfigBody = components["schemas"]["ExecutionConfigBody"];
export type ModelPolicyRef = components["schemas"]["ModelPolicyRef"];
export type LLMGatewayConfigRef = components["schemas"]["LLMGatewayConfigRef"];
export type CredentialResource = components["schemas"]["CredentialResource"];
export type CreateCredentialRequest =
  components["schemas"]["CreateCredentialRequest"];
export type RuntimeConfigAuthorDocument =
  components["schemas"]["RuntimeConfigAuthorDocument"];
export type RuntimeConfigResource =
  components["schemas"]["RuntimeConfigResource"];
type RuntimeConfigPage = components["schemas"]["RuntimeConfigPage"];
export type RuntimeLabelBinding = components["schemas"]["RuntimeLabelBinding"];
type RuntimeLabelPage = components["schemas"]["RuntimeLabelPage"];
export type RuntimeCredentialMetadata =
  components["schemas"]["RuntimeCredentialMetadata"];
export type RuntimeCredentialPage =
  components["schemas"]["RuntimeCredentialPage"];
export type CreateRuntimeCredentialRequest =
  components["schemas"]["CreateRuntimeCredentialRequest"];
export type RuntimeAgentPrincipal =
  components["schemas"]["RuntimeAgentPrincipal"];
type RuntimeAgentPrincipalPage =
  components["schemas"]["RuntimeAgentPrincipalPage"];
type SchedulerSettings = components["schemas"]["SchedulerSettings"];

export interface SchedulerSettingsSnapshot {
  resource: SchedulerSettings;
  etag: string;
}

type WritableConfigurationKind = "model-policies" | "llm-gateways";

interface CursorPageRequest {
  cursor?: string;
}

function requireConfigIdentity(name: string, version?: string): void {
  if (
    !CONFIG_NAME_PATTERN.test(name) ||
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
  if (!DIGEST.test(ref.digest)) {
    throw new TypeError("Published configuration digest is invalid");
  }
}

function requireRuntimeIdentity(name: string, version?: string): void {
  if (
    !RUNTIME_INFRASTRUCTURE_ID.test(name) ||
    (version !== undefined && !RUNTIME_CONFIG_VERSION.test(version))
  ) {
    throw new TypeError("Runtime configuration identity is invalid");
  }
}

function requireCredentialId(value: string): void {
  if (!CREDENTIAL_ID.test(value)) {
    throw new TypeError("Credential ID is invalid");
  }
}

function requireRuntimeCredentialId(value: string): void {
  if (!RUNTIME_CREDENTIAL_ID.test(value)) {
    throw new TypeError("Runtime credential ID is invalid");
  }
}

function requireRuntimeAgentId(value: string): void {
  if (!RUNTIME_AGENT_ID.test(value)) {
    throw new TypeError("Runtime Agent principal ID is invalid");
  }
}

function requireRuntimeRef(
  ref: components["schemas"]["RuntimeConfigRef"],
): components["schemas"]["RuntimeConfigRef"] {
  requireRuntimeIdentity(ref.name, ref.version);
  if (!DIGEST.test(ref.digest)) {
    throw new TypeError("RuntimeConfig ref is invalid");
  }
  return { name: ref.name, version: ref.version, digest: ref.digest };
}

/** A telemetry export never starts retrying above its maximum backoff. */
function consistentTelemetryRetry(
  telemetry: components["schemas"]["RuntimeTelemetryConfig"] | null | undefined,
): boolean {
  const retry = telemetry?.export?.retry;
  return (
    retry === undefined ||
    (retry.initialBackoffMilliseconds ??
      DEFAULT_INITIAL_BACKOFF_MILLISECONDS) <=
      (retry.maxBackoffMilliseconds ?? DEFAULT_MAX_BACKOFF_MILLISECONDS)
  );
}

/**
 * A RuntimeConfig resource is the document its ref names, and its
 * telemetry retry bounds are ordered.
 */
function requireRuntimeConfigResource(
  value: RuntimeConfigResource,
  status: number,
): RuntimeConfigResource {
  const document = value.document;
  if (
    value.ref.name !== document.metadata.name ||
    value.ref.version !== document.metadata.version ||
    !consistentTelemetryRetry(document.spec.worker?.telemetry) ||
    !consistentTelemetryRetry(document.spec.planner?.telemetry)
  ) {
    throw invalidAPIResponse(
      status,
      "Server returned an invalid RuntimeConfig resource",
    );
  }
  return value;
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

/** Workspace limits nest: one file or managed text fits the whole tree. */
function consistentWorkspace(
  value: WorkspaceCapabilities | undefined,
): boolean {
  return (
    value === undefined ||
    (value.limits.maxFileBytes <= value.limits.maxExpandedBytes &&
      value.limits.maxManagedTextBytes <= value.limits.maxExpandedBytes)
  );
}

/** A live Agent advertises each toolset once, with nesting workspace limits. */
function consistentRuntimeAgent(value: RuntimeAgentObservation): boolean {
  const toolsets = value.supportedToolsets.map((toolset) => toolset.ref);
  return (
    new Set(toolsets).size === toolsets.length &&
    consistentWorkspace(value.workspaceCapabilities)
  );
}

/**
 * A principal is offline exactly when no live Agent reports for it, and
 * mismatched exactly when it misses some of the adapters it requires.
 */
function consistentPrincipal(value: RuntimeAgentPrincipal): boolean {
  return (
    value.missingRuntimeAdapters.every((adapter) =>
      value.requiredRuntimeAdapters.includes(adapter),
    ) &&
    (value.availability === "offline") === (value.live === undefined) &&
    (value.availability === "adapter_capability_mismatch") ===
      value.missingRuntimeAdapters.length > 0 &&
    (value.live === undefined || consistentRuntimeAgent(value.live))
  );
}

/**
 * Capability lists and labels have no wire order. Sorting the freshly parsed
 * response once keeps every Operations view in the same order.
 */
function sortRuntimeAgent(value: RuntimeAgentObservation): void {
  value.supportedRuntimes.sort();
  value.supportedSandboxProfiles.sort();
  value.supportedRuntimeAdapters.sort();
  value.supportedToolsets.sort((left, right) =>
    left.ref.localeCompare(right.ref),
  );
  for (const toolset of value.supportedToolsets) toolset.tools.sort();
  value.workspaceCapabilities?.modes.sort();
}

function sortPrincipal(value: RuntimeAgentPrincipal): void {
  value.labels.sort();
  value.requiredRuntimeAdapters.sort();
  value.missingRuntimeAdapters.sort();
  if (value.live !== undefined) sortRuntimeAgent(value.live);
}

/** A model-backed allocation resolves the LLM Gateway that serves it. */
function consistentAllocation(value: AllocationObservation): boolean {
  return (
    value.executionConfig.modelPolicy === undefined ||
    value.executionConfig.llmGateway !== undefined
  );
}

function requirePrincipal(
  result: { data?: RuntimeAgentPrincipal; error?: unknown; response: Response },
  runtimeAgentId: string,
): RuntimeAgentPrincipal {
  const principal = requireData(result);
  // The Server answers for the requested principal; the ETag is its revision.
  if (
    !consistentPrincipal(principal) ||
    principal.runtimeAgentId !== runtimeAgentId ||
    result.response.headers.get("etag") !== `"${principal.revision}"`
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid Runtime Agent principal",
    );
  }
  sortPrincipal(principal);
  return principal;
}

/** Every page of an Operations list as one page; partial lists throw. */
async function collectAllPages<T>(
  load: (request: CursorPageRequest) => Promise<{
    items: T[];
    page: { hasMore: boolean; nextCursor?: string };
  }>,
  what: string,
): Promise<{ items: T[]; page: { hasMore: false } }> {
  const items = await loadCompletePages(what, (cursor) =>
    load(cursor === undefined ? {} : { cursor }),
  );
  return { items, page: { hasMore: false } };
}

function requireSchedulerSettings(
  value: SchedulerSettings,
  response: Response,
): SchedulerSettingsSnapshot {
  // The ETag is the settings revision, and the representation is never cached.
  const etag = response.headers.get("ETag");
  if (
    etag !== `"${value.revision}"` ||
    response.headers.get("Cache-Control") !== "no-store"
  ) {
    throw invalidAPIResponse(
      response.status,
      "Server returned inconsistent Scheduler settings headers",
    );
  }
  return { resource: value, etag };
}

export { listConfigurations, listCredentials };

export async function getSchedulerSettings(
  api: PublicAPI,
): Promise<SchedulerSettingsSnapshot> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/settings/scheduler"),
  );
  return requireSchedulerSettings(requireData(result), result.response);
}

export async function replaceSchedulerSettings(
  api: PublicAPI,
  maxConcurrentRuns: number,
  etag: string,
): Promise<SchedulerSettingsSnapshot> {
  if (
    !Number.isInteger(maxConcurrentRuns) ||
    maxConcurrentRuns < 1 ||
    maxConcurrentRuns > 32
  ) {
    throw new TypeError(
      "Maximum concurrent Workflow Runs must be an integer from 1 through 32",
    );
  }
  // The client checks the If-Match value against the operation's schema.
  const result = await api.request((client) =>
    client.PUT("/v1/operations/settings/scheduler", {
      params: { header: { "If-Match": etag } },
      body: { maxConcurrentRuns },
    }),
  );
  const settings = requireSchedulerSettings(
    requireData(result),
    result.response,
  );
  // The replacement answers with the value it was asked to set.
  if (settings.resource.maxConcurrentRuns !== maxConcurrentRuns) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned a different Scheduler setting after replacement",
    );
  }
  return settings;
}

export async function getOperationsSnapshot(
  api: PublicAPI,
  signal?: AbortSignal,
): Promise<OperationsSnapshot> {
  const result = await api.request((client) =>
    client.GET(
      "/v1/operations/snapshot",
      signal === undefined ? {} : { signal },
    ),
  );
  const snapshot = requireData(result);
  if (
    !snapshot.runtimeAgents.every(consistentRuntimeAgent) ||
    !snapshot.allocations.every(consistentAllocation)
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid Operations snapshot",
    );
  }
  snapshot.runtimeAgents.forEach(sortRuntimeAgent);
  return snapshot;
}

async function listRuntimeAgentPrincipals(
  api: PublicAPI,
  request: CursorPageRequest = {},
): Promise<RuntimeAgentPrincipalPage> {
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-agent-principals", {
      params: {
        query: {
          limit: OPERATIONS_PAGE_SIZE,
          ...(request.cursor === undefined ? {} : { cursor: request.cursor }),
        },
      },
    }),
  );
  const page = requireData(result);
  if (!page.items.every(consistentPrincipal)) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid Runtime Agent principal",
    );
  }
  page.items.forEach(sortPrincipal);
  return page;
}

export function listAllRuntimeAgentPrincipals(
  api: PublicAPI,
): Promise<RuntimeAgentPrincipalPage> {
  return collectAllPages(
    (request) => listRuntimeAgentPrincipals(api, request),
    "The Runtime Agent list",
  );
}

export async function getRuntimeAgentPrincipal(
  api: PublicAPI,
  runtimeAgentId: string,
): Promise<RuntimeAgentPrincipal> {
  requireRuntimeAgentId(runtimeAgentId);
  const result = await api.request((client) =>
    client.GET("/v1/operations/runtime-agent-principals/{runtimeAgentId}", {
      params: { path: { runtimeAgentId } },
    }),
  );
  return requirePrincipal(result, runtimeAgentId);
}

export async function replaceRuntimeAgentPrincipalLabels(
  api: PublicAPI,
  runtimeAgentId: string,
  labels: string[],
  revision: string,
  idempotencyKey: string,
): Promise<RuntimeAgentPrincipal> {
  requireRuntimeAgentId(runtimeAgentId);
  if (!/^[1-9][0-9]{0,19}$/.test(revision)) {
    throw new TypeError("Runtime Agent principal revision is invalid");
  }
  const normalized = [...labels].sort();
  normalized.forEach((label) => requireRuntimeIdentity(label));
  if (
    normalized.length > 32 ||
    new Set(normalized).size !== normalized.length ||
    normalized.includes("default")
  ) {
    throw new TypeError("Runtime Agent principal labels are invalid");
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
    client.PUT(
      "/v1/operations/runtime-agent-principals/{runtimeAgentId}/labels",
      {
        params: {
          path: { runtimeAgentId },
          header: {
            "Idempotency-Key": key,
            "If-Match": `"${revision}"`,
          },
        },
        body: { labels: normalized },
      },
    ),
  );
  return requirePrincipal(result, runtimeAgentId);
}

export async function deleteRuntimeAgentPrincipal(
  api: PublicAPI,
  runtimeAgentId: string,
  revision: string,
  idempotencyKey: string,
): Promise<void> {
  requireRuntimeAgentId(runtimeAgentId);
  if (!/^[1-9][0-9]{0,19}$/.test(revision)) {
    throw new TypeError("Runtime Agent principal revision is invalid");
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
    client.DELETE("/v1/operations/runtime-agent-principals/{runtimeAgentId}", {
      params: {
        path: { runtimeAgentId },
        header: {
          "Idempotency-Key": key,
          "If-Match": `"${revision}"`,
        },
      },
    }),
  );
  if (!result.response.ok) {
    throw publicAPIError(result.response.status, result.error);
  }
}

export async function getConfiguration(
  api: PublicAPI,
  kind: ConfigurationKind,
  name: string,
  version: string,
  signal?: AbortSignal,
): Promise<ConfigurationResource> {
  requireConfigIdentity(name, version);
  const result = await api.request((client) =>
    client.GET("/v1/configurations/{kind}/{name}/versions/{version}", {
      params: { path: { kind, name, version } },
      ...(signal === undefined ? {} : { signal }),
    }),
  );
  const resource = requireCanonicalConfiguration(
    requireData(result),
    result.response.status,
  );
  // The Server answers for the requested configuration version.
  if (
    resource.ref.kind !== kind ||
    resource.ref.name !== name ||
    resource.ref.version !== version
  ) {
    throw invalidAPIResponse(
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
    !CONFIG_NAME_PATTERN.test(request.name) ||
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
  const resource = requireCanonicalConfiguration(
    requireData(result),
    result.response.status,
  );
  // The publication answers with the identity it was asked to publish.
  if (
    resource.ref.kind !== kind ||
    resource.ref.name !== request.name ||
    resource.ref.version !== request.version
  ) {
    throw invalidAPIResponse(
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
  requireCredentialId(credentialId);
  const result = await api.request((client) =>
    client.GET("/v1/operations/credentials/{credentialId}", {
      params: { path: { credentialId } },
    }),
  );
  const credential = requireData(result);
  // The Server answers for the requested credential.
  if (credential.credentialId !== credentialId) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid credential resource",
    );
  }
  return credential;
}

export async function createCredential(
  api: PublicAPI,
  request: CreateCredentialRequest,
  idempotencyKey: string,
): Promise<CredentialResource> {
  requireCredentialId(request.credentialId);
  requirePublishedRef(request.llmGateway);
  request.gatewayPolicy.modelPolicies.forEach(requirePublishedRef);
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.POST("/v1/operations/credentials", {
      params: { header: { "Idempotency-Key": header } },
      body: request,
    }),
  );
  const credential = requireData(result);
  // The creation answers with the credential and Gateway it was asked for.
  if (
    credential.credentialId !== request.credentialId ||
    credential.llmGateway.gatewayId !== request.llmGateway.gatewayId ||
    credential.llmGateway.version !== request.llmGateway.version ||
    credential.llmGateway.digest !== request.llmGateway.digest
  ) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an invalid credential creation response",
    );
  }
  return credential;
}

export async function deleteCredential(
  api: PublicAPI,
  credentialId: string,
  idempotencyKey: string,
): Promise<void> {
  requireCredentialId(credentialId);
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.DELETE("/v1/operations/credentials/{credentialId}", {
      params: {
        path: { credentialId },
        header: { "Idempotency-Key": header },
      },
    }),
  );
  if (!result.response.ok) {
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
  page.items.forEach((item) =>
    requireRuntimeConfigResource(item, result.response.status),
  );
  return page;
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
  const resource = requireRuntimeConfigResource(
    requireData(result),
    result.response.status,
  );
  // The Server answers for the requested RuntimeConfig version.
  if (resource.ref.name !== name || resource.ref.version !== version) {
    throw invalidAPIResponse(
      result.response.status,
      "Server returned the wrong RuntimeConfig resource",
    );
  }
  return resource;
}

export async function publishRuntimeConfig(
  api: PublicAPI,
  document: RuntimeConfigAuthorDocument,
  idempotencyKey: string,
): Promise<RuntimeConfigResource> {
  const header = idempotencyHeader(api, idempotencyKey);
  const result = await api.request((client) =>
    client.POST("/v1/operations/runtime-configs", {
      params: { header: { "Idempotency-Key": header } },
      body: document,
    }),
  );
  const resource = requireRuntimeConfigResource(
    requireData(result),
    result.response.status,
  );
  // The publication answers with the identity it was asked to publish.
  if (
    resource.ref.name !== document.metadata.name ||
    resource.ref.version !== document.metadata.version
  ) {
    throw invalidAPIResponse(
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
  return requireData(result);
}

export function listAllRuntimeLabels(
  api: PublicAPI,
): Promise<RuntimeLabelPage> {
  return collectAllPages(
    (request) => listRuntimeLabels(api, request),
    "The Runtime label list",
  );
}

export async function putRuntimeLabel(
  api: PublicAPI,
  label: string,
  config: components["schemas"]["RuntimeConfigRef"],
  idempotencyKey: string,
  expectedRevision?: string,
): Promise<RuntimeLabelBinding> {
  requireRuntimeIdentity(label);
  const safeConfig = requireRuntimeRef(config);
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
  const binding = requireData(result);
  // The Server answers for the requested label; the ETag is its revision.
  if (
    binding.label !== label ||
    result.response.headers.get("etag") !== `"${binding.revision}"`
  ) {
    throw invalidAPIResponse(
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
  if (!result.response.ok) {
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
  return requireData(result);
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
  const metadata = requireData(result);
  // The creation answers with the credential it was asked to create.
  if (
    metadata.credentialId !== request.credentialId ||
    metadata.kind !== request.kind
  ) {
    throw invalidAPIResponse(
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
  if (!result.response.ok) {
    throw publicAPIError(result.response.status, result.error);
  }
}

export function isConfigurationKind(value: string): value is ConfigurationKind {
  return CONFIGURATION_KINDS.some((kind) => kind === value);
}
