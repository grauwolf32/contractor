import type { RunStatus } from "../../api/runs";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { AlertDialog, StatusGlyph } from "../../ui";
import { runRecoveryReason } from "../../app/vocabulary";

type Recovery = NonNullable<RunStatus["recovery"]>;

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
        <strong>{runRecoveryReason(recovery.code)}</strong>
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
  return (
    <AlertDialog
      className="runs-dialog"
      eyebrow="Model recovery"
      title="Retry the model connection?"
      tone="primary"
      cancelLabel="Not now"
      confirmLabel="Confirm retry"
      pendingLabel="Enabling retry…"
      pending={pending}
      error={error === null ? null : <ErrorNotice error={error} />}
      onConfirm={onConfirm}
      onCancel={onCancel}
    >
      <p>{runRecoveryReason(recovery.code)}</p>
      <p>
        The current invocation continues where it stopped. Completed work is
        kept and no new stage attempt starts. Restore the model before you
        retry.
      </p>
    </AlertDialog>
  );
}
