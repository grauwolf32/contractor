import type {
  AllocationObservation,
  LLMGatewayConfigRef,
  ModelPolicyRef,
} from "../../api/operations";

export interface ExactConfigurationRef {
  name: string;
  version: string;
  digest: string;
  kind: "agent-templates" | "model-policies" | "llm-gateways";
}

export function exactConfigurationRef(
  ref:
    | AllocationObservation["agentTemplate"]
    | ModelPolicyRef
    | LLMGatewayConfigRef,
): ExactConfigurationRef {
  if ("templateId" in ref) {
    return {
      kind: "agent-templates",
      name: ref.templateId,
      version: ref.version,
      digest: ref.digest,
    };
  }
  if ("policyId" in ref) {
    return {
      kind: "model-policies",
      name: ref.policyId,
      version: ref.version,
      digest: ref.digest,
    };
  }
  return {
    kind: "llm-gateways",
    name: ref.gatewayId,
    version: ref.version,
    digest: ref.digest,
  };
}
