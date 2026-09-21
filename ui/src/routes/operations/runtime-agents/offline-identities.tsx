import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  deleteRuntimeAgentPrincipal,
  type RuntimeAgentPrincipal,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { Dialog } from "../../../app/dialog";
import { Icon } from "../../../app/icon";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";
import { agentDisplayName, relativeAge, shortAgentId } from "./identity";
import { AgentLabelsDialog } from "./labels-dialog";

function ForgetIdentityDialog({
  principal,
  pending,
  error,
  onCancel,
  onConfirm,
}: {
  principal: RuntimeAgentPrincipal;
  pending: boolean;
  error: Error | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const cancel = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      className="project-dialog runtime-agent-forget-dialog panel"
      role="alertdialog"
      labelledBy={heading}
      initialFocusRef={cancel}
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Permanent action</p>
          <h2 id={heading}>Forget {agentDisplayName(principal)}?</h2>
        </div>
      </div>
      <p>
        This removes the offline identity{" "}
        <code>{principal.runtimeAgentId}</code> from Server. A process that
        registers with this ID again starts as a new identity without saved
        labels.
      </p>
      {error === null ? null : <ErrorNotice error={error} />}
      <div className="runtime-agent-dialog-actions">
        <button
          className="secondary-button"
          type="button"
          ref={cancel}
          disabled={pending}
          onClick={onCancel}
        >
          Cancel
        </button>
        <button
          className="danger-button"
          type="button"
          disabled={pending}
          onClick={onConfirm}
        >
          {pending ? "Forgetting…" : "Forget identity"}
        </button>
      </div>
    </Dialog>
  );
}

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
  const deletion = useMutation({
    mutationFn: () =>
      deleteRuntimeAgentPrincipal(
        api,
        principal.runtimeAgentId,
        principal.revision,
        `delete-runtime-principal-ui-${crypto.randomUUID()}`,
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
        Last seen{" "}
        <time
          dateTime={principal.updatedAt}
          title={formatTimestamp(principal.updatedAt)}
        >
          {relativeAge(principal.updatedAt, now)}
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
        <ForgetIdentityDialog
          principal={principal}
          pending={deletion.isPending}
          error={deletion.error}
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
