import { ErrorNotice } from "../../app/error-notice";
import { AlertDialog } from "../../ui";

/**
 * Confirmation for Continue from failed stage (S06): it names the stage and
 * warns that model and tool calls, including external side effects, repeat.
 * Nothing is sent before Confirm continuation.
 */
export function ContinueRunDialog({
  stage,
  pending,
  error,
  onConfirm,
  onCancel,
}: {
  stage: string;
  pending: boolean;
  error: Error | null;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <AlertDialog
      className="runs-dialog"
      eyebrow="Continue failed Run"
      title="Continue from failed stage?"
      tone="primary"
      cancelLabel="Back"
      confirmLabel="Confirm continuation"
      pendingLabel="Continuing…"
      pending={pending}
      error={error === null ? null : <ErrorNotice error={error} />}
      onConfirm={onConfirm}
      onCancel={onCancel}
    >
      <p>
        Retry stage <strong>{stage}</strong>? Model and tool calls will run
        again and may repeat external side effects.
      </p>
      <p>
        Successful stages and their results are preserved. The failed stage gets
        a new attempt with its saved inputs and configuration.
      </p>
    </AlertDialog>
  );
}
