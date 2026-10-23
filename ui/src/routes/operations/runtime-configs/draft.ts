// The RuntimeConfig publish form's draft and the document it publishes. Every
// grammar, bound and enum comes from the public contract (src/api/contract.ts),
// so the form checks exactly what the Server validates.
import {
  RUNTIME_CONFIG_VERSION,
  RUNTIME_INFRASTRUCTURE_ID,
  schemaEnum,
  schemaLimit,
} from "../../../api/contract";
import type { PublicSchemaName } from "../../../api/generated/response-schemas";
import type {
  ConfigurationResource,
  RuntimeConfigAuthorDocument,
} from "../../../api/operations";
import { exactKey, ruleMessage } from "./identity";

type WorkerPatch = NonNullable<RuntimeConfigAuthorDocument["spec"]["worker"]>;
type Telemetry = NonNullable<WorkerPatch["telemetry"]>;
type ProxyTarget = NonNullable<WorkerPatch["httpProxy"]>["targets"][number];

/** A new draft: every block off, the usual delivery limits filled in. */
export const emptyConfigDraft = () => ({
  name: "",
  version: "1",
  gateway: false,
  gatewayKey: "",
  llmCredential: "",
  workerTelemetry: false,
  workerTelemetryEndpoint: "",
  workerTelemetryCredential: "",
  workerTelemetryTimeout: "10",
  workerTelemetryCaptureContent: false,
  workerTelemetryBatchSize: "8",
  workerTelemetryMaxAttempts: "2",
  workerTelemetryMaxPendingSpans: "2048",
  workerTelemetryMaxPendingSize: "64",
  plannerTelemetry: false,
  plannerTelemetryEndpoint: "",
  plannerTelemetryCredential: "",
  plannerTelemetryTimeout: "10",
  plannerTelemetryCaptureContent: false,
  interactsh: false,
  interactshClear: false,
  interactshEndpoint: "",
  interactshCredential: "",
  interactshCA: "",
  interactshTimeout: "",
  httpProxy: false,
  proxyURL: "",
  proxyCredential: "",
  proxyCA: "",
  proxyTargets: [] as ProxyTarget[],
});

export type ConfigDraft = ReturnType<typeof emptyConfigDraft>;

const MiB = 1024 * 1024;

/** An integer property's bounds, in units of `unit` (bytes as MiB). */
function bounds(schema: PublicSchemaName, property: string, unit = 1) {
  return {
    minimum: Math.ceil(
      schemaLimit(schema, "minimum", "properties", property) / unit,
    ),
    maximum: Math.floor(
      schemaLimit(schema, "maximum", "properties", property) / unit,
    ),
  };
}

const FLUSH_TIMEOUT = bounds("RuntimeTelemetryConfig", "flushTimeoutSeconds");

/**
 * The numeric fields under Optional settings: label, the block that enables
 * them and their bounds. A flush timeout may stay empty (the Server default
 * applies); the export limits are always sent.
 */
export const LIMIT_FIELDS = [
  {
    field: "workerTelemetryTimeout",
    label: "Worker telemetry flush timeout (seconds)",
    block: "workerTelemetry",
    optional: true,
    ...FLUSH_TIMEOUT,
  },
  {
    field: "workerTelemetryBatchSize",
    label: "Worker telemetry batch size (MiB)",
    block: "workerTelemetry",
    optional: false,
    ...bounds("WorkerTelemetryExportConfig", "batchSizeBytes", MiB),
  },
  {
    field: "workerTelemetryMaxAttempts",
    label: "Worker telemetry maximum attempts per batch",
    block: "workerTelemetry",
    optional: false,
    ...bounds("WorkerTelemetryExportConfig", "maxAttempts"),
  },
  {
    field: "workerTelemetryMaxPendingSpans",
    label: "Worker telemetry maximum pending spans",
    block: "workerTelemetry",
    optional: false,
    ...bounds("WorkerTelemetryExportConfig", "maxPendingSpans"),
  },
  {
    field: "workerTelemetryMaxPendingSize",
    label: "Worker telemetry maximum pending size (MiB)",
    block: "workerTelemetry",
    optional: false,
    ...bounds("WorkerTelemetryExportConfig", "maxPendingBytes", MiB),
  },
  {
    field: "plannerTelemetryTimeout",
    label: "Planner telemetry flush timeout (seconds)",
    block: "plannerTelemetry",
    optional: true,
    ...FLUSH_TIMEOUT,
  },
] as const satisfies readonly {
  field: keyof ConfigDraft;
  label: string;
  block: "workerTelemetry" | "plannerTelemetry";
  optional: boolean;
  minimum: number;
  maximum: number;
}[];

const TARGET_LABELS: Record<ProxyTarget, string> = {
  "llm-gateway": "LLM Gateway HTTP",
  "tool-http": "Tool HTTP",
  "tool-subprocess": "Tool subprocess environment",
};

/** The proxy's traffic targets, in the contract's order, with their labels. */
export const PROXY_TARGETS = schemaEnum<ProxyTarget>(
  "RuntimeHTTPProxyConfig",
  "properties",
  "targets",
  "items",
).map((target) => [target, TARGET_LABELS[target]] as const);

function safeHTTPURL(value: string): boolean {
  try {
    const parsed = new URL(value);
    return (
      (parsed.protocol === "http:" || parsed.protocol === "https:") &&
      parsed.username === "" &&
      parsed.password === "" &&
      parsed.search === "" &&
      parsed.hash === ""
    );
  } catch {
    return false;
  }
}

function telemetry(
  endpoint: string,
  credential: string,
  timeout: string,
  captureContent: boolean,
): Telemetry {
  return {
    adapter: "otlp-http@1",
    endpoint,
    captureContent,
    ...(credential === "" ? {} : { credential }),
    ...(timeout === "" ? {} : { flushTimeoutSeconds: Number(timeout) }),
  };
}

function draftErrors(
  draft: ConfigDraft,
  gateway: ConfigurationResource | undefined,
): string[] {
  const errors: string[] = [];
  if (!RUNTIME_INFRASTRUCTURE_ID.test(draft.name)) {
    errors.push(ruleMessage("Name", RUNTIME_INFRASTRUCTURE_ID));
  }
  if (!RUNTIME_CONFIG_VERSION.test(draft.version)) {
    errors.push(ruleMessage("Version", RUNTIME_CONFIG_VERSION));
  }
  if (
    !draft.gateway &&
    !draft.workerTelemetry &&
    !draft.plannerTelemetry &&
    !draft.httpProxy &&
    !draft.interactsh
  ) {
    errors.push("Select at least one RuntimeConfig block.");
  }
  if (draft.gateway && gateway === undefined) {
    errors.push("Select one published LLM Gateway.");
  }
  for (const [enabled, endpoint, label] of [
    [draft.workerTelemetry, draft.workerTelemetryEndpoint, "Worker telemetry"],
    [
      draft.interactsh && !draft.interactshClear,
      draft.interactshEndpoint,
      "Interactsh",
    ],
    [
      draft.plannerTelemetry,
      draft.plannerTelemetryEndpoint,
      "Planner telemetry",
    ],
  ] as const) {
    if (enabled && !safeHTTPURL(endpoint)) {
      errors.push(
        `${label} endpoint must be an HTTP(S) URL without userinfo, query or fragment.`,
      );
    }
  }
  for (const {
    field,
    label,
    block,
    optional,
    minimum,
    maximum,
  } of LIMIT_FIELDS) {
    const value = draft[field];
    if (!draft[block] || (optional && value === "")) continue;
    const number = Number(value);
    if (!Number.isInteger(number) || number < minimum || number > maximum) {
      errors.push(`${label} must be an integer from ${minimum} to ${maximum}.`);
    }
  }
  if (
    draft.workerTelemetry &&
    Number(draft.workerTelemetryMaxPendingSize) <
      Number(draft.workerTelemetryBatchSize)
  ) {
    errors.push("Pending size must be at least the batch size.");
  }
  if (draft.httpProxy && !safeHTTPURL(draft.proxyURL)) {
    errors.push(
      "HTTP proxy URL must be HTTP(S) without userinfo, query or fragment.",
    );
  }
  if (draft.httpProxy && draft.proxyTargets.length === 0) {
    errors.push("HTTP proxy requires at least one explicit traffic target.");
  }
  if (
    draft.interactsh &&
    !draft.interactshClear &&
    draft.interactshTimeout !== ""
  ) {
    const timeout = Number(draft.interactshTimeout);
    const range = bounds("RuntimeInteractshConfig", "requestTimeoutSeconds");
    if (
      !Number.isInteger(timeout) ||
      timeout < range.minimum ||
      timeout > range.maximum
    ) {
      errors.push(
        `Interactsh timeout must be an integer from ${range.minimum} to ${range.maximum}.`,
      );
    }
  }
  return errors;
}

/**
 * The document the draft publishes, or the reasons it cannot be published.
 * `gateways` are the loaded LLM Gateway versions the draft selects from.
 */
export function buildDocument(
  draft: ConfigDraft,
  gateways: readonly ConfigurationResource[],
): { document?: RuntimeConfigAuthorDocument; errors: string[] } {
  const gateway = gateways.find(
    (resource) => exactKey(resource.ref) === draft.gatewayKey,
  );
  const errors = draftErrors(draft, gateway);
  if (errors.length > 0) return { errors };
  const worker: WorkerPatch = {
    ...(draft.interactsh
      ? {
          interactsh: draft.interactshClear
            ? null
            : {
                adapter: "interactsh-http@1" as const,
                endpoint: draft.interactshEndpoint,
                ...(draft.interactshCredential === ""
                  ? {}
                  : { credential: draft.interactshCredential }),
                ...(draft.interactshCA === ""
                  ? {}
                  : { caBundlePem: draft.interactshCA }),
                ...(draft.interactshTimeout === ""
                  ? {}
                  : { requestTimeoutSeconds: Number(draft.interactshTimeout) }),
              },
        }
      : {}),
    ...(draft.gateway && gateway !== undefined
      ? {
          llmGateway: {
            gateway: `${gateway.ref.name}@${gateway.ref.version}`,
            ...(draft.llmCredential === ""
              ? {}
              : { credential: draft.llmCredential }),
          },
        }
      : {}),
    ...(draft.workerTelemetry
      ? {
          telemetry: {
            ...telemetry(
              draft.workerTelemetryEndpoint,
              draft.workerTelemetryCredential,
              draft.workerTelemetryTimeout,
              draft.workerTelemetryCaptureContent,
            ),
            export: {
              batchSizeBytes: Number(draft.workerTelemetryBatchSize) * MiB,
              maxAttempts: Number(draft.workerTelemetryMaxAttempts),
              maxPendingSpans: Number(draft.workerTelemetryMaxPendingSpans),
              maxPendingBytes:
                Number(draft.workerTelemetryMaxPendingSize) * MiB,
            },
          },
        }
      : {}),
    ...(draft.httpProxy
      ? {
          httpProxy: {
            adapter: "http-proxy@1" as const,
            proxyUrl: draft.proxyURL,
            ...(draft.proxyCredential === ""
              ? {}
              : { credential: draft.proxyCredential }),
            ...(draft.proxyCA === "" ? {} : { caBundlePem: draft.proxyCA }),
            targets: [...draft.proxyTargets].sort(),
          },
        }
      : {}),
  };
  return {
    errors,
    document: {
      apiVersion: "contractor/v1alpha1",
      kind: "RuntimeConfig",
      metadata: { name: draft.name, version: draft.version },
      spec: {
        ...(Object.keys(worker).length === 0 ? {} : { worker }),
        ...(draft.plannerTelemetry
          ? {
              planner: {
                telemetry: telemetry(
                  draft.plannerTelemetryEndpoint,
                  draft.plannerTelemetryCredential,
                  draft.plannerTelemetryTimeout,
                  draft.plannerTelemetryCaptureContent,
                ),
              },
            }
          : {}),
      },
    },
  };
}
