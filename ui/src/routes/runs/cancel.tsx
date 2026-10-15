import { type FormEvent, useId, useRef } from "react";

import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import { Button, Field } from "../../ui";

/**
 * The Cancel Run confirmation. A cancellation needs an explicit reason; the
 * page shows the Run's state only as the Server reports it after the
 * request.
 */
export function CancelRunDialog({
  reason,
  maxLength,
  onReasonChange,
  validationError,
  pending,
  error,
  onSubmit,
  onCancel,
}: {
  reason: string;
  /** The longest reason the Server accepts. */
  maxLength: number;
  onReasonChange: (value: string) => void;
  validationError: string | undefined;
  pending: boolean;
  error: Error | null;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const field = useRef<HTMLTextAreaElement>(null);

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    onSubmit();
  }

  return (
    <Dialog
      className="ui-dialog ui-panel runs-dialog"
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
        <Field
          label="Reason"
          hint="Required. Saved with the cancellation record of this Run."
          error={validationError}
        >
          {(control) => (
            <textarea
              {...control}
              ref={field}
              name="cancellationReason"
              rows={3}
              maxLength={maxLength}
              value={reason}
              readOnly={pending}
              onChange={(event) => onReasonChange(event.target.value)}
            />
          )}
        </Field>
        {error === null ? null : <ErrorNotice error={error} />}
        <div className="ui-dialog-actions">
          <Button disabled={pending} onClick={onCancel}>
            Keep running
          </Button>
          <Button
            variant="danger"
            type="submit"
            pending={pending}
            pendingLabel="Requesting cancellation…"
          >
            Request cancellation
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
