import type { components } from "./generated/public";

const RUNTIME_ID = /^[a-z][a-z0-9_-]{0,62}$/;
const RUNTIME_VERSION = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const REVISION = /^[1-9][0-9]{0,19}$/;
const CONFIG_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const ADAPTER = /^[a-z][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9._+-]*$/;

type RunRuntimeConfiguration = components["schemas"]["RunRuntimeConfiguration"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function exactKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): void {
  const allowed = new Set([...required, ...optional]);
  if (
    required.some((key) => !(key in value)) ||
    Object.keys(value).some((key) => !allowed.has(key))
  ) {
    throw new TypeError("Runtime configuration response shape is invalid");
  }
}

function safeRef(value: unknown) {
  if (!isRecord(value)) {
    throw new TypeError("RuntimeConfig ref is invalid");
  }
  exactKeys(value, ["name", "version", "digest"]);
  if (
    typeof value.name !== "string" ||
    !RUNTIME_ID.test(value.name) ||
    typeof value.version !== "string" ||
    !RUNTIME_VERSION.test(value.version) ||
    typeof value.digest !== "string" ||
    !DIGEST.test(value.digest)
  ) {
    throw new TypeError("RuntimeConfig ref is invalid");
  }
  return {
    name: value.name,
    version: value.version,
    digest: value.digest,
  };
}

function safePin(value: unknown): RunRuntimeConfiguration["default"] {
  if (!isRecord(value)) {
    throw new TypeError("Pinned RuntimeConfig response is invalid");
  }
  exactKeys(value, ["label", "bindingRevision", "config"]);
  const config = safeRef(value.config);
  if (
    typeof value.label !== "string" ||
    !RUNTIME_ID.test(value.label) ||
    typeof value.bindingRevision !== "string" ||
    !REVISION.test(value.bindingRevision)
  ) {
    throw new TypeError("Pinned RuntimeConfig response is invalid");
  }
  return {
    label: value.label,
    bindingRevision: value.bindingRevision,
    config,
  };
}

export function safeStageRuntimeConfiguration(
  value: unknown,
): components["schemas"]["StageRuntimeConfiguration"] {
  if (!isRecord(value)) {
    throw new TypeError("Stage Runtime configuration response is invalid");
  }
  exactKeys(value, ["allocations"]);
  if (
    !Array.isArray(value.allocations) ||
    value.allocations.length === 0 ||
    value.allocations.length > 32
  ) {
    throw new TypeError("Stage Runtime allocations are invalid");
  }
  const layers = new Set([
    "default",
    "workflow_execution_config",
    "run_labels",
    "run_execution_config",
    "escalation_execution_config",
    "agent_labels",
  ]);
  const originKeys = [
    "llmGateway",
    "llmCredential",
    "workerTelemetry",
    "httpProxy",
    "plannerTelemetry",
  ] as const;
  const allocations = value.allocations.map((candidate) => {
    if (!isRecord(candidate)) {
      throw new TypeError("Stage Runtime allocation is invalid");
    }
    exactKeys(candidate, [
      "logicalAgent",
      "agentLabels",
      "runtimeAdapters",
      "origins",
      "status",
    ]);
    if (
      typeof candidate.logicalAgent !== "string" ||
      !CONFIG_ID.test(candidate.logicalAgent) ||
      !Array.isArray(candidate.agentLabels) ||
      candidate.agentLabels.length > 32 ||
      !Array.isArray(candidate.runtimeAdapters) ||
      candidate.runtimeAdapters.length > 128 ||
      !isRecord(candidate.origins) ||
      !["pinned", "release_pending", "released"].includes(
        String(candidate.status),
      )
    ) {
      throw new TypeError("Stage Runtime allocation is invalid");
    }
    const agentLabels = candidate.agentLabels.map(safePin);
    if (
      agentLabels.some(
        (pin, index) =>
          pin.label === "default" ||
          (index > 0 && agentLabels[index - 1]!.label >= pin.label),
      )
    ) {
      throw new TypeError("Stage Agent-label provenance is invalid");
    }
    const runtimeAdapters = candidate.runtimeAdapters.map((adapter) => {
      if (typeof adapter !== "string" || !ADAPTER.test(adapter)) {
        throw new TypeError("Stage Runtime adapter provenance is invalid");
      }
      return adapter;
    });
    if (
      runtimeAdapters.some(
        (adapter, index) => index > 0 && runtimeAdapters[index - 1]! >= adapter,
      )
    ) {
      throw new TypeError("Stage Runtime adapters are not canonical");
    }
    exactKeys(candidate.origins, [], originKeys);
    const origins: components["schemas"]["StageRuntimeOrigins"] = {};
    for (const key of originKeys) {
      const raw = candidate.origins[key];
      if (raw === undefined) continue;
      if (!isRecord(raw)) {
        throw new TypeError("Stage Runtime origin is invalid");
      }
      exactKeys(raw, ["layer"], ["configs"]);
      if (typeof raw.layer !== "string" || !layers.has(raw.layer)) {
        throw new TypeError("Stage Runtime origin layer is invalid");
      }
      const configs =
        raw.configs === undefined
          ? undefined
          : Array.isArray(raw.configs) && raw.configs.length <= 32
            ? raw.configs.map(safeRef)
            : (() => {
                throw new TypeError("Stage Runtime origin refs are invalid");
              })();
      const labelLayer = ["default", "run_labels", "agent_labels"].includes(
        raw.layer,
      );
      if (labelLayer !== (configs !== undefined && configs.length > 0)) {
        throw new TypeError("Stage Runtime origin refs do not match its layer");
      }
      origins[key] = {
        layer:
          raw.layer as components["schemas"]["RuntimeFieldOrigin"]["layer"],
        ...(configs === undefined ? {} : { configs }),
      };
    }
    return {
      logicalAgent: candidate.logicalAgent,
      agentLabels,
      runtimeAdapters,
      origins,
      status:
        candidate.status as components["schemas"]["StageRuntimeAllocation"]["status"],
    };
  });
  if (
    allocations.some(
      (allocation, index) =>
        index > 0 &&
        allocations[index - 1]!.logicalAgent >= allocation.logicalAgent,
    )
  ) {
    throw new TypeError("Stage Runtime allocations are not canonical");
  }
  return { allocations };
}

export function safeRunRuntimeConfiguration(
  value: unknown,
): RunRuntimeConfiguration {
  if (!isRecord(value)) {
    throw new TypeError("Run Runtime configuration response is invalid");
  }
  exactKeys(value, ["default", "labels"]);
  if (!Array.isArray(value.labels) || value.labels.length > 32) {
    throw new TypeError("Run Runtime labels are invalid");
  }
  const defaultPin = safePin(value.default);
  const labels = value.labels.map(safePin);
  if (
    defaultPin.label !== "default" ||
    labels.some((pin) => pin.label === "default") ||
    labels.some(
      (pin, index) => index > 0 && labels[index - 1]!.label >= pin.label,
    )
  ) {
    throw new TypeError("Run Runtime label order is invalid");
  }
  return { default: defaultPin, labels };
}
