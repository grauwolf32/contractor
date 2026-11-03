import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { usePublicAPI } from "../../../api/context";
import {
  deleteRuntimeAgentPrincipal,
  type RuntimeAgentPrincipal,
  type RuntimeLabelBinding,
} from "../../../api/operations";
import { queryKeys } from "../../../api/query-keys";
import { AlertDialog, Button, IdChip, StatusChip } from "../../../ui";
import { Icon } from "../../../app/icon";
import { ErrorNotice } from "../../../app/error-notice";
import { formatTimestamp, formatAge, plural } from "../../../app/format";
import { EntryHeading } from "../common";
import { agentDisplayName } from "./identity";
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
      <details className="ops-resource-row ops-resource-disclosure">
        <summary className="ops-resource-heading">
          <EntryHeading
            icon="operations"
            title={<strong>{name}</strong>}
            subtitle="No current process"
          >
            <StatusChip tone="neutral" size="sm">
              Offline
            </StatusChip>
          </EntryHeading>
        </summary>
        <div className="ops-resource-body">
          <IdChip
            value={principal.runtimeAgentId}
            label={`Agent ID for ${name}`}
          />
          <span className="ops-offline-fact">
            {labelsWereUpdated ? "Labels updated" : "Registered"}{" "}
            <time
              dateTime={factTimestamp}
              title={formatTimestamp(factTimestamp)}
            >
              {formatAge(factTimestamp, now)}
            </time>
          </span>
          <span className="ops-offline-fact">
            {plural(principal.labels.length, "label")}
          </span>
          <div className="ops-offline-actions">
            <Button
              size="xs"
              icon={<Icon name="settings" />}
              disabled={bindings === undefined || deletion.isPending}
              aria-haspopup="dialog"
              aria-label={`Edit labels for ${name}`}
              onClick={() => setEditing(true)}
            >
              Edit labels
            </Button>
            <Button
              size="xs"
              variant="danger"
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
            </Button>
          </div>
        </div>
      </details>
      {editing && bindings !== undefined ? (
        <AgentLabelsDialog
          principal={principal}
          bindings={bindings}
          onClose={() => setEditing(false)}
        />
      ) : null}
      {forgetting ? (
        <AlertDialog
          className="ops-confirm"
          eyebrow="Permanent action"
          title={<>Forget {agentDisplayName(principal)}?</>}
          confirmLabel="Forget identity"
          pendingLabel="Forgetting…"
          pending={deletion.isPending}
          error={
            deletion.error === null ? null : (
              <ErrorNotice error={deletion.error} />
            )
          }
          onCancel={() => setForgetting(false)}
          onConfirm={() => deletion.mutate()}
        >
          <p>
            This removes the offline identity{" "}
            <code>{principal.runtimeAgentId}</code> from Server. A process that
            registers with this ID again starts as a new identity without saved
            labels.
          </p>
        </AlertDialog>
      ) : null}
    </li>
  );
}

export function OfflineIdentities({
  principals,
  bindings,
  now,
}: {
  principals: RuntimeAgentPrincipal[];
  bindings: RuntimeLabelBinding[] | undefined;
  now: number;
}) {
  if (principals.length === 0) return null;
  return (
    <section
      className="ops-offline"
      role="group"
      aria-label={`Offline identities (${principals.length})`}
    >
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
    </section>
  );
}
