import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ReactNode, useId, useState } from "react";
import { useParams } from "react-router";

import { CONFIG_ID } from "../../../api/contract";
import { usePublicAPI } from "../../../api/context";
import { deleteCredential, getCredential } from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { Button, DetailHeader, Facts } from "../../../ui";
import { ConfigurationRefLink } from "../common";
import { exactConfigurationRef } from "../references";
import { ActiveChip } from "./active-chip";
import { CredentialPolicyView } from "./policy";
import { InUseErrorDetails } from "../in-use-details";
import { QueryView } from "../../../app/query-view";
import { useGuardedNavigate } from "../../../app/page-location";

/** A titled block of the credential page: eyebrow, heading and content. */
function Panel({
  eyebrow,
  title,
  className = "ops-panel",
  children,
}: {
  eyebrow: string;
  title: string;
  className?: string;
  children: ReactNode;
}) {
  const heading = useId();
  return (
    <section className={className} aria-labelledby={heading}>
      <div>
        <p className="ops-eyebrow">{eyebrow}</p>
        <h3 id={heading} className="ops-panel-title">
          {title}
        </h3>
      </div>
      {children}
    </section>
  );
}

function CredentialDeletion({ credentialId }: { credentialId: string }) {
  const api = usePublicAPI();
  const navigate = useGuardedNavigate();
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
    <Panel
      eyebrow="Permanent action"
      title="Delete credential"
      className="ops-panel ops-danger"
    >
      <p className="ui-caption">
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
        <Button
          variant="danger"
          disabled={!confirmed}
          pending={mutation.isPending}
          pendingLabel="Deleting…"
          onClick={() => mutation.mutate()}
        >
          Delete from LiteLLM and Contractor
        </Button>
      </div>
    </Panel>
  );
}

export function CredentialDetailRoute() {
  const api = usePublicAPI();
  const { credentialId = "" } = useParams();
  const valid = CONFIG_ID.test(credentialId);
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
        >
          {(data) => (
            <>
              <Facts
                className="ops-glance"
                aria-label="Credential"
                items={[
                  { term: "Safe label", value: data.label ?? "None" },
                  { term: "Created", value: formatTimestamp(data.createdAt) },
                  {
                    term: "Gateway",
                    value: (
                      <ConfigurationRefLink
                        value={exactConfigurationRef(data.llmGateway)}
                      />
                    ),
                  },
                ]}
              />
              <Panel eyebrow="Policy" title="LiteLLM-enforced limits">
                <CredentialPolicyView policy={data.effectivePolicy} />
              </Panel>
              <Panel eyebrow="Gateway observation" title="Consumption">
                {data.consumption === undefined ? (
                  <p className="ui-caption">
                    No live consumption aggregate is available.
                  </p>
                ) : (
                  <Facts
                    className="ops-glance"
                    items={[
                      { term: "Spend", value: data.consumption.spend ?? "—" },
                      {
                        term: "Requests",
                        value: data.consumption.requests ?? "—",
                      },
                      {
                        term: "Input tokens",
                        value: data.consumption.inputTokens ?? "—",
                      },
                      {
                        term: "Output tokens",
                        value: data.consumption.outputTokens ?? "—",
                      },
                      {
                        term: "Observed",
                        value:
                          data.consumption.observedAt === undefined
                            ? "—"
                            : formatTimestamp(data.consumption.observedAt),
                      },
                    ]}
                  />
                )}
              </Panel>
              <CredentialDeletion credentialId={credentialId} />
            </>
          )}
        </QueryView>
      )}
    </div>
  );
}
