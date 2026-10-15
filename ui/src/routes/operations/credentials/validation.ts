import { schemaLimit, stringRule } from "../../../api/contract";
import type { CreateCredentialRequest } from "../../../api/operations";

// CreateCredentialRequest and GatewayPolicy limits, as the contract states them.
export const LABEL = stringRule(
  "CreateCredentialRequest",
  "properties",
  "label",
);
const MAX_MODEL_POLICIES = schemaLimit(
  "GatewayPolicy",
  "maxItems",
  "properties",
  "modelPolicies",
);

/** The bounds of one integer limit of GatewayPolicy. */
export function integerBounds(
  name: "tpmLimit" | "rpmLimit" | "maxParallelRequests",
) {
  return {
    minimum: schemaLimit("GatewayPolicy", "minimum", "properties", name),
    maximum: schemaLimit("GatewayPolicy", "maximum", "properties", name),
  };
}

export function validateCredentialRequest(
  request: CreateCredentialRequest,
): string[] {
  const { gatewayPolicy: policy, label } = request;
  const errors: string[] = [];
  if (policy.modelPolicies.length === 0) {
    errors.push("Select at least one ModelPolicy.");
  }
  if (policy.modelPolicies.length > MAX_MODEL_POLICIES) {
    errors.push(`At most ${MAX_MODEL_POLICIES} ModelPolicies can be selected.`);
  }
  if (label !== undefined && !LABEL.test(label)) {
    errors.push(
      `Safe label must contain ${LABEL.minLength}–${LABEL.maxLength} characters when supplied.`,
    );
  }
  if (
    policy.maxBudget !== undefined &&
    (!Number.isFinite(policy.maxBudget) || policy.maxBudget <= 0)
  ) {
    errors.push("Maximum spend must be a finite number greater than zero.");
  }
  if (policy.budgetDuration !== undefined && policy.maxBudget === undefined) {
    errors.push("Budget reset requires a maximum spend.");
  }
  for (const [name, title] of [
    ["tpmLimit", "TPM limit"],
    ["rpmLimit", "RPM limit"],
    ["maxParallelRequests", "Parallel request limit"],
  ] as const) {
    const value = policy[name];
    const { minimum, maximum } = integerBounds(name);
    if (
      value !== undefined &&
      (!Number.isInteger(value) || value < minimum || value > maximum)
    ) {
      // V385: needs a shared integer formatter in src/app/format.ts.
      errors.push(
        `${title} must be an integer from ${minimum} through ${maximum.toLocaleString("en-US")}.`,
      );
    }
  }
  return errors;
}
