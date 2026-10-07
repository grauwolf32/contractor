import { useId, useRef } from "react";

import type { RunStatus } from "../../api/runs";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { StatusGlyph } from "../../ui";

type Recovery = NonNullable<RunStatus["recovery"]>;

const recoveryReasons: Record<Recovery["code"], string> = {
  model_unavailable: "The model was unloaded or is unavailable.",
  gateway_unavailable: "The model gateway is unavailable.",
  gateway_timeout: "The model request timed out.",
  gateway_rate_limited: "The model gateway is rate limiting requests.",
};

/**
 * Model recovery of a waiting Run, for the triage summary. The Retry model
 * connection action itself sits in the Run page header.
 */
export function RecoveryStatus({ run }: { run: RunStatus }) {
  const recovery = run.recovery;
  if (recovery === undefined) return null;
  return (
    <div className="runs-recovery" role="status">
      <StatusGlyph tone={recovery.requiresRetry ? "blocked" : "progress"} />
      <div>
        <strong>{recoveryReasons[recovery.code]}</strong>
        <p>
          {recovery.requiresRetry
            ? "Automatic recovery has paused. Restore the model, then use Retry model connection to continue."
            : recovery.nextRetryAt === undefined
              ? "Waiting for automatic recovery."
              : `Next recovery check: ${formatTimestamp(recovery.nextRetryAt)}.`}
        </p>
        <p>
          Completed work is retained. You can cancel this Run while it waits.
        </p>
      </div>
    </div>
  );
}

/**
 * Confirmation for Retry model connection: it continues the current
 * invocation, unlike Continue from failed stage (a new attempt) or Configure
 * another Run (a new Run).
 */
export function RetryModelDialog({
  recovery,
  pending,
  error,
  onConfirm,
  onCancel,
}: {
  recovery: Recovery;
  pending: boolean;
  error: Error | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const back = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      className="project-dialog panel runs-dialog"
      labelledBy={titleId}
      describedBy={descriptionId}
      initialFocusRef={back}
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
    >
      <DialogHeader
        id={titleId}
        eyebrow="Model recovery"
        title="Retry the model connection?"
      />
      <div id={descriptionId} className="runs-dialog-text">
        <p>{recoveryReasons[recovery.code]}</p>
        <p>
          The current invocation continues where it stopped. Completed work is
          kept and no new stage attempt starts. Restore the model before you
          retry.
        </p>
      </div>
      {error === null ? null : <ErrorNotice error={error} />}
      <div className="project-dialog-actions">
        <button
          className="secondary-button"
          type="button"
          ref={back}
          disabled={pending}
          onClick={onCancel}
        >
          Not now
        </button>
        <button type="button" disabled={pending} onClick={onConfirm}>
          {pending ? "Enabling retry…" : "Confirm retry"}
        </button>
      </div>
    </Dialog>
  );
}
