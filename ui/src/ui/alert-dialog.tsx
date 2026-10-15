import { useId, useRef, type ReactNode } from "react";

import { Dialog, DialogHeader } from "../app/dialog";
import { Button } from "./button";

/**
 * A confirmation that interrupts: "Delete this check?", "Continue from the
 * failed stage?". It is a modal role="alertdialog" named by its title and
 * described by `children`. Focus starts on the safe action (Cancel), Tab
 * stays inside, and focus returns to the control that opened it.
 *
 * Nothing is sent before the confirm button: `onConfirm` starts the work
 * and the caller passes `pending` while it runs. While pending, every
 * control is disabled and Escape, the backdrop and the close button do
 * nothing, so a request in flight is never abandoned by the dialog.
 */
export function AlertDialog({
  title,
  eyebrow,
  children,
  details,
  error,
  confirmLabel,
  pendingLabel,
  cancelLabel = "Cancel",
  tone = "danger",
  pending = false,
  confirmDisabled = false,
  closeLabel,
  dismissOnBackdrop = false,
  className,
  onConfirm,
  onCancel,
}: {
  title: ReactNode;
  /** A small line above the title, e.g. "Permanent workspace deletion". */
  eyebrow?: ReactNode;
  /** What happens, in words; it describes the dialog to screen readers. */
  children: ReactNode;
  /** Further content after the description: facts, a type-to-confirm field. */
  details?: ReactNode;
  /** Shown above the buttons, e.g. <ErrorNotice error={mutation.error} />. */
  error?: ReactNode;
  confirmLabel: string;
  /** The confirm button's text while pending, e.g. "Deleting…". */
  pendingLabel?: string | undefined;
  cancelLabel?: string | undefined;
  /** "danger" for destructive confirmations, "primary" for the others. */
  tone?: "danger" | "primary" | undefined;
  pending?: boolean | undefined;
  /** Disables only the confirm button, e.g. until a name is typed. */
  confirmDisabled?: boolean | undefined;
  /** Adds a close (×) button with this accessible name to the heading. */
  closeLabel?: string | undefined;
  dismissOnBackdrop?: boolean | undefined;
  /** Added to the dialog panel, for a width of its own. */
  className?: string | undefined;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const heading = useId();
  const description = useId();
  const safeAction = useRef<HTMLButtonElement>(null);
  const cancel = () => {
    if (!pending) onCancel();
  };
  return (
    <Dialog
      className={[
        "ui-dialog ui-alert-dialog ui-panel",
        tone === "danger" ? "ui-alert-dialog-danger" : undefined,
        className,
      ]
        .filter((part) => part !== undefined)
        .join(" ")}
      role="alertdialog"
      labelledBy={heading}
      describedBy={description}
      initialFocusRef={safeAction}
      dismissOnBackdrop={dismissOnBackdrop}
      onRequestClose={cancel}
    >
      <DialogHeader
        id={heading}
        title={title}
        {...(eyebrow === undefined ? {} : { eyebrow })}
        {...(closeLabel === undefined
          ? {}
          : {
              close: { label: closeLabel, disabled: pending, onClose: cancel },
            })}
      />
      <div id={description} className="ui-alert-dialog-description">
        {children}
      </div>
      {details}
      {error}
      <div className="ui-dialog-actions">
        <Button ref={safeAction} disabled={pending} onClick={cancel}>
          {cancelLabel}
        </Button>
        <Button
          variant={tone}
          pending={pending}
          pendingLabel={pendingLabel}
          disabled={confirmDisabled}
          onClick={onConfirm}
        >
          {confirmLabel}
        </Button>
      </div>
    </Dialog>
  );
}
