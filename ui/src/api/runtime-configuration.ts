import { RUNTIME_INFRASTRUCTURE_ID } from "./contract";
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

/** The label of the base binding, which no Run selects (RCF-R118). */
const DEFAULT_RUNTIME_LABEL = "default";

/** Runtime labels one Run selects and one Runtime Agent holds (LIM-RCF-26). */
export const RUNTIME_LABEL_LIMIT = 32;

/**
 * A Runtime label set in its canonical lexical order, or undefined when the
 * labels are not one: at most RUNTIME_LABEL_LIMIT labels of the label
 * grammar, none of them default and none twice (RCF-R117 to RCF-R120).
 * Request order has no meaning. The shared cases of
 * api/testdata/v1alpha1/runtime-label-set-cases.json hold the Server and the
 * Runtime to the same verdicts.
 */
export function canonicalRuntimeLabels(
  labels: readonly string[],
): string[] | undefined {
  if (labels.length > RUNTIME_LABEL_LIMIT) return undefined;
  const canonical = [...labels].sort();
  return canonical.every(
    (label, index) =>
      RUNTIME_INFRASTRUCTURE_ID.test(label) &&
      label !== DEFAULT_RUNTIME_LABEL &&
      (index === 0 || canonical[index - 1] !== label),
  )
    ? canonical
    : undefined;
}

/** Label pins are canonical: a Runtime label set already in canonical order. */
function canonicalLabelPins(pins: readonly PinnedRuntimeConfig[]): boolean {
  const labels = pins.map((pin) => pin.label);
  const canonical = canonicalRuntimeLabels(labels);
  return (
    canonical !== undefined &&
    canonical.every((label, index) => label === labels[index])
  );
}

/**
 * Invariants of a Run's Runtime pins that the schema cannot express: the
 * default pin carries the default label and the label pins are canonical.
 */
export function canonicalRunRuntimeConfiguration(
  value: RunRuntimeConfiguration,
): boolean {
  return (
    value.default.label === DEFAULT_RUNTIME_LABEL &&
    canonicalLabelPins(value.labels)
  );
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
