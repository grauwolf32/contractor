import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState } from "react";
import {
  auditMutationAudit,
  mutateAudit,
  type Audit,
  type AuditMutationAction,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { PublicAPIError } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
import { Dialog } from "../../../app/dialog";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { ErrorNotice } from "../../artifacts/common";

export function AuditMutationNotice({ error }: { error: unknown }) {
  return (
    <>
      <ErrorNotice error={error} reconcileWrite />
      {error instanceof PublicAPIError && error.status === 412 ? (
        <p className="muted-copy" role="status">
          The Audit revision changed. The page has refreshed authoritative
          state; review it before retrying the action.
        </p>
      ) : null}
    </>
  );
}

type DestructiveAuditAction = Extract<AuditMutationAction, "cancel" | "delete">;

function auditActionAllowed(
  audit: Audit,
  action: DestructiveAuditAction,
): boolean {
  if (action === "cancel") {
    return (
      audit.state === "active" ||
      audit.state === "waiting_review" ||
      audit.state === "paused" ||
      audit.state === "finalizing"
    );
  }
  return (
    audit.state === "draft" ||
    audit.state === "completed" ||
    audit.state === "cancelled" ||
    audit.state === "failed"
  );
}

function AuditMutationDialog({
  action,
  audit,
  projectName,
  error,
  pending,
  onClose,
  onConfirm,
}: {
  action: DestructiveAuditAction;
  audit: Audit;
  projectName: string | undefined;
  error: unknown;
  pending: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const description = useId();
  const safeAction = useRef<HTMLButtonElement>(null);
  const allowed = auditActionAllowed(audit, action);
  const cancelling = action === "cancel";
  return (
    <Dialog
      className="project-dialog panel audit-mutation-dialog"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={safeAction}
      onRequestClose={onClose}
      role="alertdialog"
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Audit action</p>
          <h2 id={heading}>
            {cancelling ? "Cancel this Audit?" : "Delete this Audit?"}
          </h2>
        </div>
        <button
          className="project-dialog-close"
          type="button"
          aria-label="Close Audit confirmation"
          disabled={pending}
          onClick={onClose}
        >
          ×
        </button>
      </div>
      <p id={description}>
        {cancelling
          ? "Stop this audit and cancel its running checks. Results already collected will remain available. Cancellation may take a moment."
          : "Permanently delete this audit, its results and retained evidence. This cannot be undone. Deletion runs in the background and may take a moment."}
      </p>
      <dl className="metadata-grid audit-mutation-identity">
        <div>
          <dt>Project</dt>
          <dd>
            {projectName ?? audit.projectId} <code>{audit.projectId}</code>
          </dd>
        </div>
        <div>
          <dt>Audit</dt>
          <dd>
            <code>{audit.auditId}</code>
          </dd>
        </div>
        <div>
          <dt>Profile</dt>
          <dd>
            <code>
              {audit.profile.name}@{audit.profile.version}
            </code>
          </dd>
        </div>
        <div>
          <dt>Current state</dt>
          <dd>
            {audit.state} · revision {audit.revision}
          </dd>
        </div>
      </dl>
      {!allowed ? (
        <div className="notice notice-warning" role="status">
          <strong>This action is no longer available.</strong>
          <p>
            The audit is now <code>{audit.state}</code>. Close this confirmation
            and review its updated status.
          </p>
        </div>
      ) : null}
      {error === null ? null : <AuditMutationNotice error={error} />}
      <div className="inline-actions audit-mutation-actions">
        <button
          ref={safeAction}
          type="button"
          className="secondary-button"
          disabled={pending}
          onClick={onClose}
        >
          Keep Audit unchanged
        </button>
        <button
          type="button"
          className="danger-button"
          disabled={pending || !allowed}
          onClick={onConfirm}
        >
          {pending
            ? cancelling
              ? "Cancelling…"
              : "Starting deletion…"
            : cancelling
              ? "Confirm cancellation"
              : "Begin Audit deletion"}
        </button>
      </div>
    </Dialog>
  );
}

export function AuditControls({
  audit,
  projectName,
  deleteOnly = false,
}: {
  audit: Audit;
  projectName: string | undefined;
  deleteOnly?: boolean;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [confirmation, setConfirmation] = useState<DestructiveAuditAction>();
  const destructiveRequestInFlight = useRef(false);
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        action: AuditMutationAction;
        auditId: string;
        revision: number;
      }>("mutate-audit"),
  );
  const mutation = useMutation({
    mutationFn: (action: AuditMutationAction) => {
      const draft = {
        action,
        auditId: audit.auditId,
        revision: audit.revision,
      };
      return mutateAudit(api, action, {
        auditId: audit.auditId,
        expectedRevision: audit.revision,
        idempotencyKey: keyring.keyFor(draft),
      });
    },
    onSuccess: async (result) => {
      setConfirmation(undefined);
      const updated = auditMutationAudit(result);
      queryClient.setQueryData(
        queryKeys.audits.detail(updated.auditId),
        updated,
      );
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.projects.audits.all(updated.projectId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.allItems(updated.auditId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.allCoverage(updated.auditId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.report(updated.auditId),
        }),
      ]);
    },
    onError: async () => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.detail(audit.auditId),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.projects.audits.all(audit.projectId),
        }),
      ]);
    },
    onSettled: () => {
      destructiveRequestInFlight.current = false;
    },
  });
  function closeConfirmation(): void {
    if (mutation.isPending) return;
    mutation.reset();
    setConfirmation(undefined);
  }
  function confirmDestructiveAction(): void {
    if (
      confirmation === undefined ||
      mutation.isPending ||
      destructiveRequestInFlight.current ||
      !auditActionAllowed(audit, confirmation)
    ) {
      return;
    }
    destructiveRequestInFlight.current = true;
    mutation.reset();
    mutation.mutate(confirmation);
  }
  const buttons: Array<{
    action: AuditMutationAction;
    label: string;
    dangerous?: boolean;
  }> = [];
  if (audit.state === "draft")
    buttons.push({ action: "start", label: "Start Audit" });
  if (audit.state === "active" || audit.state === "waiting_review") {
    buttons.push({ action: "pause", label: "Pause new Runs" });
  }
  if (audit.state === "paused")
    buttons.push({ action: "resume", label: "Resume" });
  if (
    audit.state === "active" ||
    audit.state === "waiting_review" ||
    audit.state === "paused" ||
    audit.state === "finalizing"
  ) {
    buttons.push({ action: "cancel", label: "Cancel", dangerous: true });
  }
  if (
    audit.state === "draft" ||
    audit.state === "completed" ||
    audit.state === "cancelled" ||
    audit.state === "failed"
  ) {
    buttons.push({ action: "delete", label: "Delete Audit", dangerous: true });
  }
  return (
    <div className="audit-controls">
      {buttons
        .filter((button) => !deleteOnly || button.action === "delete")
        .map((button) => (
          <button
            key={button.action}
            className={button.dangerous ? "danger-button" : "secondary-button"}
            type="button"
            disabled={mutation.isPending}
            onClick={() => {
              if (button.dangerous) {
                mutation.reset();
                setConfirmation(button.action as DestructiveAuditAction);
              } else {
                mutation.mutate(button.action);
              }
            }}
          >
            {mutation.isPending && mutation.variables === button.action
              ? `${button.label}…`
              : button.label}
          </button>
        ))}
      {!auditActionAllowed(audit, "delete") ? (
        <div className="audit-delete-hint">
          <button className="danger-button" type="button" disabled>
            Delete Audit
          </button>
          <small>
            {audit.state === "deleting"
              ? "Deleting audit and retained results…"
              : audit.state === "cancelling"
                ? "Waiting for cancellation to finish."
                : "Cancel or finish the audit before deleting it."}
          </small>
        </div>
      ) : null}
      {confirmation === undefined && mutation.error !== null ? (
        <div className="audit-control-error">
          <AuditMutationNotice error={mutation.error} />
        </div>
      ) : null}
      {confirmation === undefined ? null : (
        <AuditMutationDialog
          action={confirmation}
          audit={audit}
          projectName={projectName}
          error={mutation.error}
          pending={mutation.isPending}
          onClose={closeConfirmation}
          onConfirm={confirmDestructiveAction}
        />
      )}
    </div>
  );
}
