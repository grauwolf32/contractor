import { describe, expect, it } from "vitest";

import type { CreateCredentialRequest } from "../../../api/operations";
import { validateCredentialRequest } from "./validation";

const digest = `sha256:${"1".repeat(64)}`;

function request(): CreateCredentialRequest {
  return {
    credentialId: "worker-budget",
    llmGateway: { gatewayId: "local", version: "1", digest },
    gatewayPolicy: {
      modelPolicies: [{ policyId: "worker", version: "1", digest }],
      maxBudget: 10,
      budgetDuration: "1d",
      tpmLimit: 1000,
    },
  };
}

describe("Credential policy validation", () => {
  it("accepts bounded typed LiteLLM policy without a credential value", () => {
    const value = request();
    expect(validateCredentialRequest(value)).toEqual([]);
    expect(JSON.stringify(value)).not.toMatch(/token|secret|remoteKey/i);
  });

  it("requires a policy set and valid dependent/rate values", () => {
    const value = request();
    value.gatewayPolicy.modelPolicies = [];
    delete value.gatewayPolicy.maxBudget;
    value.gatewayPolicy.tpmLimit = 0;
    expect(validateCredentialRequest(value)).toEqual([
      "Select at least one exact ModelPolicy.",
      "Budget reset requires a maximum spend.",
      "TPM limit must be an integer from 1 through 2,147,483,647.",
    ]);
  });
});
