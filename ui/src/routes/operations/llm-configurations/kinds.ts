import type { ConfigurationKind } from "../../../api/operations";

/** How each configuration kind is named on the page. */
export const KIND_LABELS: Readonly<
  Record<ConfigurationKind, { singular: string; plural: string }>
> = {
  "model-policies": { singular: "Model policy", plural: "Model policies" },
  "llm-gateways": { singular: "LLM gateway", plural: "LLM gateways" },
  "agent-templates": { singular: "Agent template", plural: "Agent templates" },
  "execution-configs": {
    singular: "Execution config",
    plural: "Execution configs",
  },
};

/** Writable kinds first; the read-only ones follow. */
export const KIND_ORDER: readonly ConfigurationKind[] = [
  "model-policies",
  "llm-gateways",
  "agent-templates",
  "execution-configs",
];

/** The configuration list, filtered to one kind (model policies by default). */
export function configurationListPath(kind: ConfigurationKind): string {
  return kind === "model-policies"
    ? "/operations/configurations"
    : `/operations/configurations?kind=${kind}`;
}
