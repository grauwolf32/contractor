import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { deleteCredential, getCredential } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CONFIG_ID_PATTERN } from "../../../api/workflows";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { ConfigurationRefLink } from "../common";
import { exactConfigurationRef } from "../references";
import { CredentialPolicyView } from "./policy";
import { InUseErrorDetails } from "../in-use-details";
import { QueryView } from "../../../app/query-view";

function CredentialDeletion({ credentialId }: { credentialId: string }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [confirmed, setConfirmed] = useState(false);
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{ credentialId: string }>("delete-credential"),
  );
  const mutation = useMutation({
    mutationFn: async () => {
      const request = { credentialId };
      await deleteCredential(api, credentialId, keyring.keyFor(request));
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.credentials.all }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.snapshot,
        }),
      ]);
      await navigate("/operations/credentials");
    },
    onError: async () => {
      await queryClient.invalidateQueries({
        queryKey: queryKeys.credentials.all,
      });
    },
  });
  return (
    <div className="panel credential-delete-panel">
      <p className="eyebrow">Permanent action</p>
      <h3>Delete credential</h3>
      <p className="muted-copy">
        Deletion is rejected while a non-terminal Run, Audit dispatch hold, or
        active RuntimeConfig label references this credential. No disable,
        update, rotation, or force-delete shortcut exists.
      </p>
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(event) => setConfirmed(event.target.checked)}
        />
        I understand that the LiteLLM key and encrypted Contractor record will
        be removed.
      </label>
      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
      <InUseErrorDetails error={mutation.error} />
      <button
        className="danger-button"
        type="button"
        disabled={!confirmed || mutation.isPending}
        onClick={() => mutation.mutate()}
      >
        {mutation.isPending
          ? "Deleting…"
          : "Delete from LiteLLM and Contractor"}
      </button>
    </div>
  );
}

export function CredentialDetailRoute() {
  const api = usePublicAPI();
  const { credentialId = "" } = useParams();
  const valid = CONFIG_ID_PATTERN.test(credentialId);
  const query = useQuery({
    queryKey: queryKeys.credentials.detail(credentialId),
    queryFn: () => getCredential(api, credentialId),
    enabled: valid,
  });
  return (
    <div className="credential-detail">
      <Link className="back-link" to="/operations/credentials">
        ← Active credentials
      </Link>
      <p className="eyebrow">Credential</p>
      <h3>{credentialId}</h3>
      {!valid ? (
        <ErrorNotice error={new Error("Credential route is invalid")} />
      ) : (
        <QueryView
          query={query}
          loading={
            <p className="loading-copy" role="status">
              Loading credential metadata…
            </p>
          }
          onRetry={() => void query.refetch()}
        >
          {(data) => (
            <>
              <dl className="metadata-grid panel">
                <div>
                  <dt>Status</dt>
                  <dd>
                    <span className="state-badge state-succeeded">active</span>
                  </dd>
                </div>
                <div>
                  <dt>Safe label</dt>
                  <dd>{data.label ?? "None"}</dd>
                </div>
                <div>
                  <dt>Created</dt>
                  <dd>{formatTimestamp(data.createdAt)}</dd>
                </div>
                <div>
                  <dt>Gateway</dt>
                  <dd>
                    <ConfigurationRefLink
                      value={exactConfigurationRef(data.llmGateway)}
                    />
                  </dd>
                </div>
              </dl>
              <div className="panel credential-policy-panel">
                <p className="eyebrow">Policy</p>
                <h3>LiteLLM-enforced limits</h3>
                <CredentialPolicyView policy={data.effectivePolicy} />
              </div>
              <div className="panel credential-consumption-panel">
                <p className="eyebrow">Gateway observation</p>
                <h3>Consumption</h3>
                {data.consumption === undefined ? (
                  <p className="compact-empty">
                    No live consumption aggregate is available.
                  </p>
                ) : (
                  <dl className="metrics-grid">
                    <div>
                      <dt>Spend</dt>
                      <dd>{data.consumption.spend ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Requests</dt>
                      <dd>{data.consumption.requests ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Input tokens</dt>
                      <dd>{data.consumption.inputTokens ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Output tokens</dt>
                      <dd>{data.consumption.outputTokens ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Observed</dt>
                      <dd>
                        {data.consumption.observedAt === undefined
                          ? "—"
                          : formatTimestamp(data.consumption.observedAt)}
                      </dd>
                    </div>
                  </dl>
                )}
              </div>
              <CredentialDeletion credentialId={credentialId} />
            </>
          )}
        </QueryView>
      )}
    </div>
  );
}
