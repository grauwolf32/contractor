import { type FormEvent, useId, useRef } from "react";

import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";

export const CANCELLATION_REASON_LIMIT = 4096;

/**
 * The Cancel Run confirmation. A cancellation needs an explicit reason
 * (1–4096 characters); the page shows the Run's state only as the Server
 * reports it after the request.
 */
export function CancelRunDialog({
  reason,
  onReasonChange,
  validationError,
  pending,
  error,
  onSubmit,
  onCancel,
}: {
  reason: string;
  onReasonChange: (value: string) => void;
  validationError: string | undefined;
  pending: boolean;
  error: Error | null;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const fieldId = useId();
  const hintId = useId();
  const errorId = useId();
  const field = useRef<HTMLTextAreaElement>(null);

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    onSubmit();
  }

  return (
    <Dialog
      className="project-dialog panel runs-dialog"
      labelledBy={titleId}
      describedBy={descriptionId}
      initialFocusRef={field}
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
    >
      <DialogHeader id={titleId} title="Cancel this Run?" />
      <p id={descriptionId} className="runs-dialog-text">
        Active work stops and Scheduler cleans up. The Run shows Cancelling
        until the Server reports that cleanup has finished; it is never shown as
        cancelled before that.
      </p>
      <form className="runs-dialog-form" onSubmit={submit} noValidate>
        <label htmlFor={fieldId}>Reason</label>
        <p className="runs-hint" id={hintId}>
          Required. Saved with the cancellation record of this Run.
        </p>
        <textarea
          id={fieldId}
          ref={field}
          name="cancellationReason"
          rows={3}
          maxLength={CANCELLATION_REASON_LIMIT}
          value={reason}
          readOnly={pending}
          aria-invalid={validationError === undefined ? undefined : true}
          aria-describedby={
            validationError === undefined ? hintId : `${hintId} ${errorId}`
          }
          onChange={(event) => onReasonChange(event.target.value)}
        />
        {validationError === undefined ? null : (
          <p className="runs-field-error" id={errorId} role="alert">
            {validationError}
          </p>
        )}
        {error === null ? null : <ErrorNotice error={error} />}
        <div className="project-dialog-actions">
          <button
            className="secondary-button"
            type="button"
            disabled={pending}
            onClick={onCancel}
          >
            Keep running
          </button>
          <button className="danger-button" type="submit" disabled={pending}>
            {pending ? "Requesting cancellation…" : "Request cancellation"}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
