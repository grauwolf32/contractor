import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { listCredentials } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../../artifacts/common";
import { CredentialCreateForm } from "./form";

export function CredentialListRoute() {
  const api = usePublicAPI();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.credentials.list(cursor),
    queryFn: () => listCredentials(api, cursor === undefined ? {} : { cursor }),
  });
  return (
    <>
      <div className="panel operations-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Secret-free metadata</p>
            <h3>Active credentials</h3>
            <p className="muted-copy">
              Every listed row is active. A disabled key must be removed from
              LiteLLM and from Contractor rather than hidden behind UI state.
            </p>
          </div>
        </div>
        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading credentials…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            No active credential metadata exists.
          </div>
        ) : (
          <div className="table-scroll">
            <table>
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
                {query.data.items.map((credential) => (
                  <tr key={credential.credentialId}>
                    <td>
                      <Link
                        to={`/operations/credentials/${encodeURIComponent(credential.credentialId)}`}
                      >
                        {credential.credentialId}
                      </Link>
                      <small className="reconciled">active</small>
                      {credential.label === undefined ? null : (
                        <span>{credential.label}</span>
                      )}
                    </td>
                    <td>
                      <code>
                        {credential.llmGateway.gatewayId}@
                        {credential.llmGateway.version}
                      </code>
                    </td>
                    <td>
                      {credential.effectivePolicy.modelPolicies.length} exact
                    </td>
                    <td>{credential.consumption?.spend ?? "not observed"}</td>
                    <td>{formatTimestamp(credential.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Credential pages"
          canGoBack={cursors.length > 1}
          {...(query.data?.page.hasMore === true &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={() =>
            setCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setCursors((current) => [...current, next])}
        />
      </div>
      <CredentialCreateForm />
    </>
  );
}
