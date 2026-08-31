import type { CreateCredentialRequest } from "../../../api/operations";

export function validateCredentialRequest(
  request: CreateCredentialRequest,
): string[] {
  const errors: string[] = [];
  if (request.gatewayPolicy.modelPolicies.length === 0) {
    errors.push("Select at least one exact ModelPolicy.");
  }
  if (request.gatewayPolicy.modelPolicies.length > 128) {
    errors.push("At most 128 exact ModelPolicies can be selected.");
  }
  if (
    request.label !== undefined &&
    (request.label.length === 0 || request.label.length > 256)
  ) {
    errors.push("Safe label must contain 1–256 characters when supplied.");
  }
  if (
    request.gatewayPolicy.maxBudget !== undefined &&
    (!Number.isFinite(request.gatewayPolicy.maxBudget) ||
      request.gatewayPolicy.maxBudget <= 0)
  ) {
    errors.push("Maximum spend must be a finite number greater than zero.");
  }
  if (
    request.gatewayPolicy.budgetDuration !== undefined &&
    request.gatewayPolicy.maxBudget === undefined
  ) {
    errors.push("Budget reset requires a maximum spend.");
  }
  const integerLimits = [
    ["TPM limit", request.gatewayPolicy.tpmLimit],
    ["RPM limit", request.gatewayPolicy.rpmLimit],
    ["Parallel request limit", request.gatewayPolicy.maxParallelRequests],
  ] as const;
  for (const [label, value] of integerLimits) {
    if (
      value !== undefined &&
      (!Number.isInteger(value) || value < 1 || value > 2_147_483_647)
    ) {
      errors.push(`${label} must be an integer from 1 through 2,147,483,647.`);
    }
  }
  return errors;
}
