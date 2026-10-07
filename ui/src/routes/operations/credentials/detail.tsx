import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { useNavigate, useParams } from "react-router";

import { usePublicAPI } from "../../../api/context";
import { deleteCredential, getCredential } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { CONFIG_ID_PATTERN } from "../../../api/workflows";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { DetailHeader } from "../../../ui";
import { ConfigurationRefLink, Glance } from "../common";
import { exactConfigurationRef } from "../references";
import { ActiveChip } from "./active-chip";
import { CredentialPolicyView } from "./policy";
import { InUseErrorDetails } from "../in-use-details";
import { QueryView } from "../../../app/query-view";

function CredentialDeletion({ credentialId }: { credentialId: string }) {
  const api = usePublicAPI();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const heading = useId();
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
    <section className="ops-panel ops-danger" aria-labelledby={heading}>
      <div>
        <p className="ops-eyebrow">Permanent action</p>
        <h3 id={heading} className="ops-panel-title">
          Delete credential
        </h3>
      </div>
      <p className="ops-note">
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
      <div className="ops-danger-actions">
        <button
          className="ui-btn"
          data-variant="danger"
          type="button"
          disabled={!confirmed || mutation.isPending}
          onClick={() => mutation.mutate()}
        >
          {mutation.isPending
            ? "Deleting…"
            : "Delete from LiteLLM and Contractor"}
        </button>
      </div>
    </section>
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
    <div className="ops-stack ops-detail">
      <DetailHeader
        breadcrumb={[{ label: "Credentials", to: "/operations/credentials" }]}
        title={credentialId}
        status={query.data === undefined ? undefined : <ActiveChip />}
      />
      {!valid ? (
        <ErrorNotice error={new Error("Credential route is invalid")} />
      ) : (
        <QueryView
          query={query}
          loading={
            <p className="ops-loading" role="status">
              Loading credential metadata…
            </p>
          }
          onRetry={() => void query.refetch()}
        >
          {(data) => (
            <>
              <Glance
                label="Credential"
                items={[
                  ["Safe label", data.label ?? "None"],
                  ["Created", formatTimestamp(data.createdAt)],
                  [
                    "Gateway",
                    <ConfigurationRefLink
                      key="gateway"
                      value={exactConfigurationRef(data.llmGateway)}
                    />,
                  ],
                ]}
              />
              <section
                className="ops-panel"
                aria-labelledby="credential-policy-heading"
              >
                <div>
                  <p className="ops-eyebrow">Policy</p>
                  <h3
                    id="credential-policy-heading"
                    className="ops-panel-title"
                  >
                    LiteLLM-enforced limits
                  </h3>
                </div>
                <CredentialPolicyView policy={data.effectivePolicy} />
              </section>
              <section
                className="ops-panel"
                aria-labelledby="credential-consumption-heading"
              >
                <div>
                  <p className="ops-eyebrow">Gateway observation</p>
                  <h3
                    id="credential-consumption-heading"
                    className="ops-panel-title"
                  >
                    Consumption
                  </h3>
                </div>
                {data.consumption === undefined ? (
                  <p className="ops-note">
                    No live consumption aggregate is available.
                  </p>
                ) : (
                  <Glance
                    items={[
                      ["Spend", data.consumption.spend ?? "—"],
                      ["Requests", data.consumption.requests ?? "—"],
                      ["Input tokens", data.consumption.inputTokens ?? "—"],
                      ["Output tokens", data.consumption.outputTokens ?? "—"],
                      [
                        "Observed",
                        data.consumption.observedAt === undefined
                          ? "—"
                          : formatTimestamp(data.consumption.observedAt),
                      ],
                    ]}
                  />
                )}
              </section>
              <CredentialDeletion credentialId={credentialId} />
            </>
          )}
        </QueryView>
      )}
    </div>
  );
}
