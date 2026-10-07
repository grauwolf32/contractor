import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { listCredentials } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { RUNTIME_CONFIGURATION_PATH } from "../../../app/navigation";
import { CursorControls } from "../../../app/cursor-controls";
import { useCursorStack } from "../../../app/pagination";
import { DisclosureChevron, OpsSection, ScopeChip } from "../common";
import { SETTINGS_PATH } from "../settings/path";
import { ActiveChip } from "./active-chip";
import { CredentialCreateForm } from "./form";
import { RecordedTime } from "../../../app/recorded-time";
import { QueryView } from "../../../app/query-view";

export function CredentialListRoute() {
  const api = usePublicAPI();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.credentials.list(cursor),
    queryFn: () => listCredentials(api, cursor === undefined ? {} : { cursor }),
  });
  return (
    <div className="ops-stack">
      <OpsSection
        id="managed-credentials-heading"
        eyebrow="LLM gateway access"
        title="Managed LLM credentials"
        description={
          <>
            <p>
              Gateway credentials created in Contractor, with their budgets and
              usage. Development tokens configured outside the UI are not listed
              here.
            </p>
            <p>
              <Link to={RUNTIME_CONFIGURATION_PATH}>
                Runtime service credentials
              </Link>
              {" are managed under Configuration; "}
              <Link to={SETTINGS_PATH}>Git SSH keys</Link>
              {" are in Settings."}
            </p>
          </>
        }
        actions={<ScopeChip>Server-wide</ScopeChip>}
      >
        <QueryView
          query={query}
          loading={
            <p className="ops-loading" role="status">
              Loading credentials…
            </p>
          }
          onRetry={() => void query.refetch()}
          isEmpty={(data) => data.items.length === 0}
          empty={
            <div className="ops-empty">
              <strong>No managed LLM credentials</strong>
              <p>
                Use Create LLM credential below to manage gateway access and
                budgets here.
              </p>
            </div>
          }
        >
          {(data) => (
            <div className="ops-table-wrap">
              <table className="ops-table" data-stack="">
                <thead>
                  <tr>
                    <th>Credential</th>
                    <th>Gateway</th>
                    <th>Policy set</th>
                    <th>Spend</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((credential) => (
                    <tr key={credential.credentialId}>
                      <td data-label="Credential">
                        <span className="ops-chips">
                          <Link
                            className="ops-mono"
                            to={`/operations/credentials/${encodeURIComponent(credential.credentialId)}`}
                          >
                            {credential.credentialId}
                          </Link>
                          <ActiveChip />
                        </span>
                        {credential.label === undefined ? null : (
                          <span className="ops-cell-sub">
                            {credential.label}
                          </span>
                        )}
                      </td>
                      <td data-label="Gateway">
                        <code className="ops-mono">
                          {credential.llmGateway.gatewayId}@
                          {credential.llmGateway.version}
                        </code>
                      </td>
                      <td data-label="Policy set">
                        {credential.effectivePolicy.modelPolicies.length}
                      </td>
                      <td data-label="Spend">
                        {credential.consumption?.spend ?? (
                          <span className="ops-muted">not observed</span>
                        )}
                      </td>
                      <td data-label="Created">
                        <RecordedTime value={credential.createdAt} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </QueryView>
        <CursorControls
          label="Credential pages"
          {...pages.controls(query.data?.page)}
        />
      </OpsSection>
      <details className="ops-disclosure configuration-clone">
        <summary>
          <DisclosureChevron />
          Create LLM credential
        </summary>
        <CredentialCreateForm />
      </details>
    </div>
  );
}
