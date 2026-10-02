import { useId, useRef, type ReactNode } from "react";

import { Dialog } from "./dialog";

export function ConfirmRemovalDialog({
  title,
  description,
  confirmLabel,
  pending,
  confirmDisabled = false,
  error,
  onCancel,
  onConfirm,
}: {
  title: string;
  description: ReactNode;
  confirmLabel: string;
  pending: boolean;
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
      className="project-dialog panel"
      role="alertdialog"
      labelledBy={heading}
      describedBy={details}
      initialFocusRef={cancel}
      dismissOnBackdrop
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Confirm removal</p>
          <h2 id={heading}>{title}</h2>
        </div>
      </div>
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
          {pending ? "Removing…" : confirmLabel}
        </button>
      </div>
    </Dialog>
  );
}
