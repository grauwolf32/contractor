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
import { IdChip } from "../../../ui";
import { agentDisplayName, relativeAge } from "./identity";
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
  return (
    <li className="ops-offline-row">
      <div className="ops-offline-name">
        <span className="ops-dot" data-offline="" aria-hidden="true" />
        <strong>{name}</strong>
        <IdChip
          value={principal.runtimeAgentId}
          label={`Agent ID for ${name}`}
        />
      </div>
      <span className="ops-offline-fact">
        {labelsWereUpdated ? "Labels updated" : "Registered"}{" "}
        <time dateTime={factTimestamp} title={formatTimestamp(factTimestamp)}>
          {relativeAge(factTimestamp, now)}
        </time>
      </span>
      <span className="ops-offline-fact">
        {principal.labels.length}{" "}
        {principal.labels.length === 1 ? "label" : "labels"}
      </span>
      <div className="ops-offline-actions">
        <button
          className="ui-btn"
          data-size="xs"
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
          className="ui-btn"
          data-size="xs"
          data-variant="danger"
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
          className="ops-confirm"
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
      className="ops-offline"
      open={open || undefined}
      aria-labelledby={heading}
    >
      <summary id={heading}>
        <svg
          className="ui-tech-chevron"
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          focusable="false"
        >
          <path d="M9.5 6l6 6-6 6" />
        </svg>
        Offline identities ({principals.length})
      </summary>
      <p className="ops-offline-note">
        No live process is registered for these IDs. Saved labels are retained
        for the next registration; an identity without labels can be forgotten.
      </p>
      <ul className="ops-offline-list">
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
