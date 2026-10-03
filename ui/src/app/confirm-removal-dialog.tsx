import { useId, useRef, type ReactNode } from "react";

import { Dialog, DialogHeader } from "./dialog";

export function ConfirmRemovalDialog({
  title,
  description,
  confirmLabel,
  pending,
  eyebrow = "Confirm removal",
  pendingLabel = "Removing…",
  className,
  dismissOnBackdrop = true,
  confirmDisabled = false,
  error,
  onCancel,
  onConfirm,
}: {
  title: ReactNode;
  description: ReactNode;
  confirmLabel: string;
  pending: boolean;
  eyebrow?: string;
  pendingLabel?: string;
  /** Extra class for the dialog panel. */
  className?: string;
  dismissOnBackdrop?: boolean;
  confirmDisabled?: boolean;
  error?: ReactNode;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const details = useId();
  const cancel = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      className={
        className === undefined
          ? "project-dialog panel"
          : `project-dialog ${className} panel`
      }
      role="alertdialog"
      labelledBy={heading}
      describedBy={details}
      initialFocusRef={cancel}
      dismissOnBackdrop={dismissOnBackdrop}
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
    >
      <DialogHeader id={heading} eyebrow={eyebrow} title={title} />
      <p id={details}>{description}</p>
      {error}
      <div className="project-dialog-actions">
        <button
          className="secondary-button"
          type="button"
          ref={cancel}
          disabled={pending}
          onClick={onCancel}
        >
          Cancel
        </button>
        <button
          className="danger-button"
          type="button"
          disabled={pending || confirmDisabled}
          onClick={onConfirm}
        >
          {pending ? pendingLabel : confirmLabel}
        </button>
      </div>
    </Dialog>
  );
}
