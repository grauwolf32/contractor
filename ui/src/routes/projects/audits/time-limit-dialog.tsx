import { useId, useRef, useState } from "react";

import type { Audit } from "../../../api/audits";
import { Dialog, DialogHeader } from "../../../app/dialog";
import { AuditMutationNotice } from "./controls";

import "./checks.css";

/**
 * Asks for the time limit before a check starts or continues: 7 days (the
 * default), 24 hours, no time limit or a custom duration up to 365 days. A
 * paused check that still has time left keeps it by default. Closing the
 * dialog sends nothing.
 */
export function AuditTimeLimitDialog({
  audit,
  action,
  pending,
  error,
  onClose,
  onConfirm,
}: {
  audit: Audit;
  action: "start" | "resume";
  pending: boolean;
  error: unknown;
  onClose: () => void;
  onConfirm: (seconds: number | undefined) => void;
}) {
  const heading = useId();
  const initialFocus = useRef<HTMLSelectElement>(null);
  const canKeepTime =
    action === "resume" &&
    audit.state === "paused" &&
    audit.stopReason?.code !== "deadline_exhausted";
  const [choice, setChoice] = useState(canKeepTime ? "remaining" : "604800");
  const [hours, setHours] = useState("168");
  const seconds =
    choice === "remaining"
      ? undefined
      : choice === "custom"
        ? Math.round(Number(hours) * 3600)
        : Number(choice);
  const valid =
    seconds === undefined ||
    (Number.isSafeInteger(seconds) &&
      seconds >= (choice === "custom" ? 1 : 0) &&
      seconds <= 31536000);
  const title = action === "start" ? "Start check" : "Continue check";
  return (
    <Dialog
      className="project-dialog panel checks-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={() => {
        if (!pending) onClose();
      }}
    >
      <form
        className="checks-dialog-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (valid && !pending) onConfirm(seconds);
        }}
      >
        <DialogHeader
          id={heading}
          title={title}
          close={{
            label: "Close time limit settings",
            disabled: pending,
            onClose: onClose,
          }}
        />
        <p className="checks-dialog-text">
          {action === "start"
            ? "Choose how long this check may start new work."
            : "Continue with the same inputs and accepted results. Finished items stay finished; running work can finish."}
        </p>
        <label className="checks-field">
          <span>Time limit</span>
          <select
            ref={initialFocus}
            value={choice}
            disabled={pending}
            onChange={(event) => setChoice(event.target.value)}
          >
            {canKeepTime ? (
              <option value="remaining">Keep remaining time</option>
            ) : null}
            <option value="604800">7 days</option>
            <option value="86400">24 hours</option>
            <option value="0">No time limit</option>
            <option value="custom">Custom duration</option>
          </select>
        </label>
        {choice === "custom" ? (
          <label className="checks-field">
            <span>Time limit in hours</span>
            <input
              type="number"
              min="0.01"
              max="8760"
              step="0.01"
              required
              value={hours}
              disabled={pending}
              onChange={(event) => setHours(event.target.value)}
            />
          </label>
        ) : null}
        <p className="checks-quiet">
          The time includes waiting in the queue and stops while the check is
          paused. Reaching the limit pauses new work; running work can finish.
        </p>
        {!valid ? (
          <p className="form-error" role="alert">
            Enter a duration greater than zero and no longer than 365 days.
          </p>
        ) : null}
        {error === null ? null : <AuditMutationNotice error={error} />}
        <div className="checks-dialog-actions">
          <button
            type="button"
            className="ui-btn"
            disabled={pending}
            onClick={onClose}
          >
            Close
          </button>
          <button
            type="submit"
            className="ui-btn"
            data-variant="primary"
            disabled={pending || !valid}
          >
            {pending ? "Applying…" : title}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
