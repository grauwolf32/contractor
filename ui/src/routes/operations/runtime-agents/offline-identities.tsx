import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  deleteRuntimeAgentPrincipal,
  type RuntimeAgentPrincipal,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { ConfirmRemovalDialog } from "../../../app/confirm-removal-dialog";
import { Icon } from "../../../app/icon";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp } from "../../../app/format";
import { agentDisplayName, relativeAge, shortAgentId } from "./identity";
import { AgentLabelsDialog } from "./labels-dialog";
import { createMutationIdempotencyKey } from "../../../mutations/idempotency";

function OfflineIdentityRow({
  principal,
  bindings,
  now,
}: {
  principal: RuntimeAgentPrincipal;
  bindings: RuntimeLabelBinding[] | undefined;
  now: number;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [forgetting, setForgetting] = useState(false);
  const [copyStatus, setCopyStatus] = useState<string>();
  const name = agentDisplayName(principal);
  const hasLabels = principal.labels.length > 0;
  const labelsWereUpdated = principal.updatedAt !== principal.createdAt;
  const factTimestamp = labelsWereUpdated
    ? principal.updatedAt
    : principal.createdAt;
  const deletion = useMutation({
    mutationFn: () =>
      deleteRuntimeAgentPrincipal(
        api,
        principal.runtimeAgentId,
        principal.revision,
        createMutationIdempotencyKey("delete-runtime-principal"),
      ),
    onSuccess: async () => {
      setForgetting(false);
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.snapshot,
        }),
      ]);
    },
    onError: () =>
      queryClient.invalidateQueries({
        queryKey: queryKeys.operations.runtimeAgentPrincipals.all,
      }),
  });
  async function copyId() {
    try {
      await navigator.clipboard.writeText(principal.runtimeAgentId);
      setCopyStatus("Agent ID copied.");
    } catch {
      setCopyStatus("Copy unavailable.");
    }
  }
  return (
    <li className="runtime-offline-identity">
      <div className="runtime-offline-identity-name">
        <span className="runtime-agent-dot is-offline" aria-hidden="true" />
        <strong>{name}</strong>
        <code
          className="runtime-agent-short-id"
          title={principal.runtimeAgentId}
        >
          {shortAgentId(principal.runtimeAgentId)}
        </code>
        <button
          className="runtime-agent-copy"
          type="button"
          title={`Copy Agent ID for ${name}`}
          aria-label={`Copy Agent ID for ${name}`}
          onClick={() => void copyId()}
        >
          <Icon name="copy" />
        </button>
        {copyStatus === undefined ? null : (
          <span className="runtime-agent-copy-status" role="status">
            {copyStatus}
          </span>
        )}
      </div>
      <span className="runtime-offline-identity-fact">
        {labelsWereUpdated ? "Labels updated" : "Registered"}{" "}
        <time dateTime={factTimestamp} title={formatTimestamp(factTimestamp)}>
          {relativeAge(factTimestamp, now)}
        </time>
      </span>
      <span className="runtime-offline-identity-fact">
        {principal.labels.length}{" "}
        {principal.labels.length === 1 ? "label" : "labels"}
      </span>
      <div className="runtime-offline-identity-actions">
        <button
          className="runtime-agent-edit"
          type="button"
          disabled={bindings === undefined || deletion.isPending}
          aria-haspopup="dialog"
          aria-label={`Edit labels for ${name}`}
          onClick={() => setEditing(true)}
        >
          <Icon name="settings" />
          Edit labels
        </button>
        <button
          className="runtime-agent-forget"
          type="button"
          disabled={hasLabels || deletion.isPending}
          aria-haspopup="dialog"
          aria-label={`Forget ${name}`}
          title={
            hasLabels
              ? "Clear the saved labels before forgetting this identity."
              : "Forget this offline identity"
          }
          onClick={() => {
            deletion.reset();
            setForgetting(true);
          }}
        >
          Forget
        </button>
      </div>
      {editing && bindings !== undefined ? (
        <AgentLabelsDialog
          principal={principal}
          bindings={bindings}
          onClose={() => setEditing(false)}
        />
      ) : null}
      {forgetting ? (
        <ConfirmRemovalDialog
          className="runtime-agent-forget-dialog"
          eyebrow="Permanent action"
          title={<>Forget {agentDisplayName(principal)}?</>}
          description={
            <>
              This removes the offline identity{" "}
              <code>{principal.runtimeAgentId}</code> from Server. A process
              that registers with this ID again starts as a new identity without
              saved labels.
            </>
          }
          confirmLabel="Forget identity"
          pendingLabel="Forgetting…"
          pending={deletion.isPending}
          dismissOnBackdrop={false}
          error={
            deletion.error === null ? null : (
              <ErrorNotice error={deletion.error} />
            )
          }
          onCancel={() => setForgetting(false)}
          onConfirm={() => deletion.mutate()}
        />
      ) : null}
    </li>
  );
}

export function OfflineIdentities({
  principals,
  bindings,
  now,
  open,
}: {
  principals: RuntimeAgentPrincipal[];
  bindings: RuntimeLabelBinding[] | undefined;
  now: number;
  open: boolean;
}) {
  const heading = useId();
  if (principals.length === 0) return null;
  return (
    <details
      className="runtime-offline-identities"
      open={open || undefined}
      aria-labelledby={heading}
    >
      <summary id={heading}>Offline identities ({principals.length})</summary>
      <p className="runtime-agent-offline-note">
        No live process is registered for these IDs. Saved labels are retained
        for the next registration; an identity without labels can be forgotten.
      </p>
      <ul className="runtime-offline-identity-list">
        {principals.map((principal) => (
          <OfflineIdentityRow
            key={principal.runtimeAgentId}
            principal={principal}
            bindings={bindings}
            now={now}
          />
        ))}
      </ul>
    </details>
  );
}
