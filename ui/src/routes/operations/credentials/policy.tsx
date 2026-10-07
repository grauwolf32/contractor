import type { CredentialResource } from "../../../api/operations";
import { ConfigurationRefLink } from "../common";
import { exactConfigurationRef } from "../references";

type EffectivePolicy = CredentialResource["effectivePolicy"];

export function CredentialPolicyView({ policy }: { policy: EffectivePolicy }) {
  const gatewayLimits = [
    ["Maximum spend", policy.maxBudget],
    ["Budget reset", policy.budgetDuration],
    ["TPM limit", policy.tpmLimit],
    ["RPM limit", policy.rpmLimit],
    ["Parallel request limit", policy.maxParallelRequests],
  ] as const;
  return (
    <div className="ops-policy">
      <div className="ops-grid-2">
        <div>
          <h4 className="ops-policy-title">Allowed ModelPolicies</h4>
          <ul className="ops-value-list">
            {policy.modelPolicies.map((ref) => (
              <li key={`${ref.policyId}@${ref.version}:${ref.digest}`}>
                <ConfigurationRefLink value={exactConfigurationRef(ref)} />
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4 className="ops-policy-title">Resolved Gateway model aliases</h4>
          <ul className="ops-value-list">
            {policy.models.map((model) => (
              <li key={model}>
                <code className="ops-label-chip">{model}</code>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <dl
        className="ops-glance"
        aria-label="Limits enforced by the LiteLLM Gateway"
      >
        {gatewayLimits.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>
              <code>{value ?? "not set"}</code>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
