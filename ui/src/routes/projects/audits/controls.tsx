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
import { ActionMenu } from "../../../app/action-menu";
import { Dialog } from "../../../app/dialog";
import { Icon } from "../../../app/icon";
import { AuditTimeLimitDialog } from "./time-limit-dialog";
import { DeleteIcon } from "../../../app/delete-icon";
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
  compact = false,
  menu = false,
}: {
  audit: Audit;
  projectName: string | undefined;
  compact?: boolean;
  /** Header layout: Delete lives in an overflow menu next to the controls. */
  menu?: boolean;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [confirmation, setConfirmation] = useState<DestructiveAuditAction>();
  const destructiveRequestInFlight = useRef(false);
  const [timeAction, setTimeAction] = useState<"start" | "resume">();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<{
        action: AuditMutationAction;
        auditId: string;
        revision: number;
        deadlineSeconds?: number;
      }>("mutate-audit"),
  );
  const mutation = useMutation({
    mutationFn: ({
      action,
      deadlineSeconds,
    }: {
      action: AuditMutationAction;
      deadlineSeconds?: number;
    }) => {
      const draft = {
        action,
        auditId: audit.auditId,
        revision: audit.revision,
        ...(deadlineSeconds === undefined ? {} : { deadlineSeconds }),
      };
      return mutateAudit(api, action, {
        auditId: audit.auditId,
        expectedRevision: audit.revision,
        idempotencyKey: keyring.keyFor(draft),
        ...(deadlineSeconds === undefined ? {} : { deadlineSeconds }),
      });
    },
    onSuccess: async (result) => {
      setConfirmation(undefined);
      setTimeAction(undefined);
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
    mutation.mutate({ action: confirmation });
  }
  const buttons: Array<{
    action: AuditMutationAction;
    label: string;
    dangerous?: boolean;
  }> = [];
  if (audit.state === "draft")
    buttons.push({ action: "start", label: "Start Audit" });
  if (audit.state === "active" || audit.state === "waiting_review") {
    buttons.push({ action: "pause", label: "Pause new Audit Runs" });
  }
  if (audit.state === "paused")
    buttons.push({ action: "resume", label: "Continue Audit" });
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
  function renderButton(button: (typeof buttons)[number], inMenu = false) {
    return (
      <button
        key={button.action}
        className={
          button.action === "delete"
            ? inMenu
              ? "danger-button"
              : "danger-button delete-icon-button"
            : button.dangerous
              ? `danger-button ${compact ? "icon-button" : ""}`
              : (button.action === "start" || button.action === "resume") &&
                  !compact
                ? "primary-button"
                : "secondary-button icon-button"
        }
        type="button"
        aria-label={inMenu ? undefined : button.label}
        title={
          button.action === "pause"
            ? "Pause new Audit Runs; running work can finish"
            : button.label
        }
        aria-haspopup={button.action === "pause" ? undefined : "dialog"}
        disabled={mutation.isPending}
        onClick={() => {
          if (button.dangerous) {
            mutation.reset();
            setConfirmation(button.action as DestructiveAuditAction);
          } else if (button.action === "start" || button.action === "resume") {
            mutation.reset();
            setTimeAction(button.action);
          } else {
            mutation.mutate({ action: button.action });
          }
        }}
      >
        {button.action === "delete" ? (
          <>
            <DeleteIcon />
            {inMenu ? <span>{button.label}</span> : null}
          </>
        ) : button.action === "pause" ? (
          <Icon name="pause" />
        ) : button.action === "start" || button.action === "resume" ? (
          <>
            <Icon name="play" />
            {compact ? null : <span>{button.label}</span>}
          </>
        ) : compact ? (
          <Icon name="stop" />
        ) : (
          button.label
        )}
      </button>
    );
  }
  const deleteButton = buttons.find((button) => button.action === "delete");
  const deleteHint =
    audit.state === "deleting"
      ? "Deleting audit and retained results…"
      : audit.state === "cancelling"
        ? "Waiting for cancellation to finish."
        : "Cancel or finish the audit before deleting it.";
  return (
    <div
      className={`audit-controls ${menu ? "audit-header-actions" : ""}`.trim()}
      id={compact ? undefined : "audit-controls"}
    >
      {!compact &&
      audit.state === "paused" &&
      audit.stopReason?.code === "deadline_exhausted" ? (
        <p className="audit-control-reason">
          <strong>Time limit reached.</strong> Continue with a longer limit;
          collected results are retained.
        </p>
      ) : null}
      {buttons
        .filter((button) => !(menu && button.action === "delete"))
        .map((button) => renderButton(button))}
      {menu ? (
        <ActionMenu label="Audit actions">
          {deleteButton === undefined ? (
            <>
              <button className="danger-button" type="button" disabled>
                <DeleteIcon />
                <span>Delete Audit</span>
              </button>
              <small>{deleteHint}</small>
            </>
          ) : (
            renderButton(deleteButton, true)
          )}
        </ActionMenu>
      ) : null}
      {!menu && !auditActionAllowed(audit, "delete") ? (
        <div className="audit-delete-hint">
          <button
            className="danger-button delete-icon-button"
            type="button"
            aria-label="Delete Audit"
            title="Delete Audit"
            disabled
          >
            <DeleteIcon />
          </button>
          {compact ? null : <small>{deleteHint}</small>}
        </div>
      ) : null}
      {confirmation === undefined &&
      timeAction === undefined &&
      mutation.error !== null ? (
        <div className="audit-control-error">
          <AuditMutationNotice error={mutation.error} />
        </div>
      ) : null}
      {timeAction === undefined ? null : (
        <AuditTimeLimitDialog
          audit={audit}
          action={timeAction}
          pending={mutation.isPending}
          error={mutation.error}
          onClose={() => {
            if (!mutation.isPending) {
              mutation.reset();
              setTimeAction(undefined);
            }
          }}
          onConfirm={(seconds) => {
            if (!mutation.isPending)
              mutation.mutate({
                action: timeAction,
                ...(seconds === undefined ? {} : { deadlineSeconds: seconds }),
              });
          }}
        />
      )}
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
