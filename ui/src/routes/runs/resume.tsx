import { useId, useRef } from "react";

import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";

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
  const titleId = useId();
  const descriptionId = useId();
  const back = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      className="project-dialog panel runs-dialog"
      role="alertdialog"
      labelledBy={titleId}
      describedBy={descriptionId}
      initialFocusRef={back}
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
    >
      <DialogHeader
        id={titleId}
        eyebrow="Continue failed Run"
        title="Continue from failed stage?"
      />
      <div id={descriptionId} className="runs-dialog-text">
        <p>
          Retry stage <strong>{stage}</strong>? Model and tool calls will run
          again and may repeat external side effects.
        </p>
        <p>
          Successful stages and their results are preserved. The failed stage
          gets a new attempt with its saved inputs and configuration.
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
          Back
        </button>
        <button type="button" disabled={pending} onClick={onConfirm}>
          {pending ? "Continuing…" : "Confirm continuation"}
        </button>
      </div>
    </Dialog>
  );
}
