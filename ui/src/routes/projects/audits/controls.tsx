import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useId, useRef, useState, type ReactNode } from "react";

import {
  auditMutationAudit,
  mutateAudit,
  type Audit,
  type AuditMutationAction,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidateCrossProject } from "../../../api/cross-project";
import { PublicAPIError } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
import { ActionMenu } from "../../../app/action-menu";
import { DeleteIcon } from "../../../app/delete-icon";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { ErrorNotice } from "../../../app/error-notice";
import { Icon } from "../../../app/icon";
import { checkStateLabel } from "../../../app/vocabulary";
import { MutationDraftKeyring } from "../../../mutations/idempotency";
import { checkKeys } from "./check-data";
import { auditProfileLabel } from "./labels";
import { AuditTimeLimitDialog } from "./time-limit-dialog";

import "./checks.css";

export function AuditMutationNotice({ error }: { error: unknown }) {
  return (
    <>
      <ErrorNotice error={error} reconcileWrite />
      {error instanceof PublicAPIError && error.status === 412 ? (
        <p className="checks-quiet" role="status">
          The check changed in the meantime. The page now shows its current
          state; review it before trying again.
        </p>
      ) : null}
    </>
  );
}

type DestructiveAuditAction = Extract<AuditMutationAction, "cancel" | "delete">;

/** States that accept Stop (cancel). */
const STOPPABLE: readonly Audit["state"][] = [
  "active",
  "waiting_review",
  "paused",
  "finalizing",
];

/** States that accept Delete. */
const DELETABLE: readonly Audit["state"][] = [
  "draft",
  "completed",
  "cancelled",
  "failed",
];

function auditActionAllowed(
  audit: Audit,
  action: DestructiveAuditAction,
): boolean {
  return (action === "cancel" ? STOPPABLE : DELETABLE).includes(audit.state);
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
  const stopping = action === "cancel";
  const state = checkStateLabel(audit.state).label;
  return (
    <Dialog
      className="project-dialog panel checks-dialog"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={safeAction}
      onRequestClose={onClose}
      role="alertdialog"
    >
      <DialogHeader
        id={heading}
        title={stopping ? "Stop this check?" : "Delete this check?"}
        close={{
          label: "Close confirmation",
          disabled: pending,
          onClose: onClose,
        }}
      />
      <p id={description} className="checks-dialog-text">
        {stopping
          ? "Stop this check and cancel its running work. Results already collected stay available. Stopping may take a moment."
          : "Permanently delete this check, its results and retained evidence. This cannot be undone. Deletion runs in the background and may take a moment."}
      </p>
      <dl className="checks-facts checks-dialog-facts">
        <div>
          <dt>Project</dt>
          <dd>
            {projectName ?? audit.projectId}{" "}
            <code className="checks-mono">{audit.projectId}</code>
          </dd>
        </div>
        <div>
          <dt>Check</dt>
          <dd>
            <code className="checks-mono">{audit.auditId}</code>
          </dd>
        </div>
        <div>
          <dt>Check type</dt>
          <dd>
            {auditProfileLabel(audit)}{" "}
            <code className="checks-mono">
              {audit.profile.name}@{audit.profile.version}
            </code>
          </dd>
        </div>
        <div>
          <dt>Current state</dt>
          <dd>
            {state} · revision {audit.revision}
          </dd>
        </div>
      </dl>
      {!allowed ? (
        <div className="checks-notice" data-tone="warning" role="status">
          <strong>This action is no longer available.</strong>
          <p>
            The check is now {state.toLocaleLowerCase()}. Close this
            confirmation and review its current state.
          </p>
        </div>
      ) : null}
      {error === null ? null : <AuditMutationNotice error={error} />}
      <div className="checks-dialog-actions">
        <button
          ref={safeAction}
          type="button"
          className="ui-btn"
          disabled={pending}
          onClick={onClose}
        >
          {stopping ? "Keep the check running" : "Keep the check"}
        </button>
        <button
          type="button"
          className="ui-btn"
          data-variant="danger"
          disabled={pending || !allowed}
          onClick={onConfirm}
        >
          {pending
            ? stopping
              ? "Stopping…"
              : "Deleting…"
            : stopping
              ? "Stop check"
              : "Delete check"}
        </button>
      </div>
    </Dialog>
  );
}

interface ControlButton {
  action: AuditMutationAction;
  /** Visible text. */
  text: string;
  /** Accessible name; starts with the visible text. */
  name: string;
  title: string;
  icon: ReactNode;
  variant: "primary" | "secondary" | "ghost" | "danger";
}

function controlButtons(audit: Audit): ControlButton[] {
  const buttons: ControlButton[] = [];
  if (audit.state === "draft")
    buttons.push({
      action: "start",
      text: "Start check",
      name: "Start check",
      title: "Start check",
      icon: <Icon name="play" />,
      variant: "primary",
    });
  if (audit.state === "active" || audit.state === "waiting_review")
    buttons.push({
      action: "pause",
      text: "Pause",
      name: "Pause new work",
      title: "Pause new work; running work can finish",
      icon: <Icon name="pause" />,
      variant: "secondary",
    });
  if (audit.state === "paused")
    buttons.push({
      action: "resume",
      text: "Continue",
      name: "Continue check",
      title: "Continue check",
      icon: <Icon name="play" />,
      variant: "primary",
    });
  if (STOPPABLE.includes(audit.state))
    buttons.push({
      action: "cancel",
      text: "Stop",
      name: "Stop check",
      title: "Stop check",
      icon: <Icon name="stop" />,
      variant: "ghost",
    });
  if (DELETABLE.includes(audit.state))
    buttons.push({
      action: "delete",
      text: "Delete check",
      name: "Delete check",
      title: "Delete check",
      icon: <DeleteIcon />,
      variant: "danger",
    });
  return buttons;
}

function deleteHint(audit: Audit): string {
  if (audit.state === "deleting")
    return "Deleting the check and its retained results…";
  if (audit.state === "cancelling") return "Waiting for the check to stop.";
  return "Stop the check or let it finish before deleting it.";
}

/**
 * Start, Pause new work, Continue, Stop and Delete for one check, as its
 * state allows. Start and Continue ask for a time limit; Stop and Delete ask
 * for confirmation. Nothing changes before the Server answers; the check,
 * its project lists and the cross-project lists are read again afterwards.
 *
 * `compact`: icon buttons for list rows. `menu`: Delete moves into a "Check
 * actions" menu next to the other controls (headers).
 */
export function AuditControls({
  audit,
  projectName,
  compact = false,
  menu = false,
}: {
  audit: Audit;
  projectName: string | undefined;
  compact?: boolean;
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
      // An in-flight detail fetch started before the mutation would otherwise
      // overwrite the newer revision when it resolves.
      await queryClient.cancelQueries({
        queryKey: queryKeys.audits.detail(updated.auditId),
        exact: true,
      });
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
          // URL-pinned coverage belongs to its original revision.
          predicate: (query) => query.queryKey.at(-1) === null,
        }),
        // The current counts; list rows read theirs when the list sees the
        // new revision.
        queryClient.invalidateQueries({
          queryKey: checkKeys.workspace(updated.auditId),
          exact: true,
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.audits.report(updated.auditId),
        }),
        invalidateCrossProject(queryClient),
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
        invalidateCrossProject(queryClient),
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
  function activate(button: ControlButton): void {
    if (button.action === "cancel" || button.action === "delete") {
      mutation.reset();
      setConfirmation(button.action);
    } else if (button.action === "start" || button.action === "resume") {
      mutation.reset();
      setTimeAction(button.action);
    } else {
      mutation.mutate({ action: button.action });
    }
  }
  const buttons = controlButtons(audit);
  function renderButton(button: ControlButton, inMenu = false) {
    const iconOnly = compact && !inMenu;
    return (
      <button
        key={button.action}
        type="button"
        className="ui-btn"
        data-size="sm"
        data-variant={
          iconOnly && button.variant === "ghost" ? "secondary" : button.variant
        }
        data-icon-only={iconOnly ? "" : undefined}
        aria-label={
          iconOnly || button.name !== button.text ? button.name : undefined
        }
        title={button.title}
        aria-haspopup={button.action === "pause" ? undefined : "dialog"}
        disabled={mutation.isPending}
        onClick={() => activate(button)}
      >
        {button.icon}
        {iconOnly ? null : <span>{button.text}</span>}
      </button>
    );
  }
  const deleteButton = buttons.find((button) => button.action === "delete");
  const visible = buttons.filter(
    (button) => !(menu && button.action === "delete"),
  );
  return (
    <div
      className="checks-controls"
      data-layout={compact ? "compact" : menu ? "menu" : undefined}
    >
      {visible.map((button) => renderButton(button))}
      {menu ? (
        <ActionMenu label="Check actions">
          {deleteButton === undefined ? (
            <>
              <button
                className="ui-btn"
                data-size="sm"
                data-variant="danger"
                type="button"
                disabled
              >
                <DeleteIcon />
                <span>Delete check</span>
              </button>
              <small className="checks-quiet">{deleteHint(audit)}</small>
            </>
          ) : (
            renderButton(deleteButton, true)
          )}
        </ActionMenu>
      ) : null}
      {!menu && deleteButton === undefined ? (
        <button
          className="ui-btn"
          data-size="sm"
          data-variant="danger"
          data-icon-only={compact ? "" : undefined}
          type="button"
          aria-label="Delete check"
          title={deleteHint(audit)}
          disabled
        >
          <DeleteIcon />
          {compact ? null : <span>Delete check</span>}
        </button>
      ) : null}
      {confirmation === undefined &&
      timeAction === undefined &&
      mutation.error !== null ? (
        <div className="checks-controls-error">
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
