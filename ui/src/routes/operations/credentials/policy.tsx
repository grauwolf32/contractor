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
    <div className="credential-policy-grid">
      <div>
        <h4>Allowed exact ModelPolicies</h4>
        <ul className="compact-value-list">
          {policy.modelPolicies.map((ref) => (
            <li key={`${ref.policyId}@${ref.version}:${ref.digest}`}>
              <ConfigurationRefLink value={exactConfigurationRef(ref)} />
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h4>Resolved Gateway model aliases</h4>
        <ul className="compact-value-list">
          {policy.models.map((model) => (
            <li key={model}>
              <code>{model}</code>
            </li>
          ))}
        </ul>
      </div>
      <dl className="key-value-list">
        {gatewayLimits.map(([label, value]) => (
          <div key={label}>
            <dt>{label} · Gateway-enforced (LiteLLM)</dt>
            <dd>
              <code>{value ?? "not set"}</code>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
