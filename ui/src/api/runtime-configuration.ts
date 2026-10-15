import type { components } from "./generated/public";

type PinnedRuntimeConfig = components["schemas"]["PinnedRuntimeConfig"];
type RunRuntimeConfiguration = components["schemas"]["RunRuntimeConfiguration"];
type StageRuntimeConfiguration =
  components["schemas"]["StageRuntimeConfiguration"];

// Origin layers that resolve through label bindings and so name their configs.
const LABEL_LAYERS = new Set(["default", "run_labels", "agent_labels"]);

/** Whether `values` are strictly ascending, the Server's canonical order. */
export function canonicalOrder(values: readonly string[]): boolean {
  return values.every(
    (value, index) => index === 0 || values[index - 1]! < value,
  );
}

/** Label pins are canonical: ordered by label and never the default label. */
function canonicalLabelPins(pins: readonly PinnedRuntimeConfig[]): boolean {
  return (
    canonicalOrder(pins.map((pin) => pin.label)) &&
    pins.every((pin) => pin.label !== "default")
  );
}

/**
 * Invariants of a Run's Runtime pins that the schema cannot express: the
 * default pin carries the default label and the label pins are canonical.
 */
export function canonicalRunRuntimeConfiguration(
  value: RunRuntimeConfiguration,
): boolean {
  return value.default.label === "default" && canonicalLabelPins(value.labels);
}

/**
 * Invariants of a Stage's allocation-pinned projection that the schema
 * cannot express: allocations, Agent labels and adapters are in canonical
 * order, and an origin names configs exactly when a label layer set it.
 */
export function canonicalStageRuntimeConfiguration(
  value: StageRuntimeConfiguration,
): boolean {
  return (
    canonicalOrder(
      value.allocations.map((allocation) => allocation.logicalAgent),
    ) &&
    value.allocations.every(
      (allocation) =>
        canonicalLabelPins(allocation.agentLabels) &&
        canonicalOrder(allocation.runtimeAdapters) &&
        Object.values(allocation.origins).every(
          (origin) =>
            LABEL_LAYERS.has(origin.layer) ===
            (origin.configs !== undefined && origin.configs.length > 0),
        ),
    )
  );
}
