import { useId, useRef, useState } from "react";
import type { Audit } from "../../../api/audits";
import { Dialog } from "../../../app/dialog";
import { AuditMutationNotice } from "./controls";

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
  const title = action === "start" ? "Start Audit" : "Continue Audit";
  return (
    <Dialog
      className="project-dialog panel audit-time-dialog"
      labelledBy={heading}
      initialFocusRef={initialFocus}
      onRequestClose={() => {
        if (!pending) onClose();
      }}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (valid && !pending) onConfirm(seconds);
        }}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Execution time</p>
            <h2 id={heading}>{title}</h2>
          </div>
          <button
            type="button"
            className="project-dialog-close"
            aria-label="Close Audit time settings"
            disabled={pending}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p>
          {action === "start"
            ? "Choose how long this audit may submit new Runs."
            : "Continue with the same inputs and accepted results. Only unfinished work with attempts remaining will run; previous reports and attempt history are retained."}
        </p>
        <label>
          Audit time limit
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
          <label>
            Time limit in hours
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
        <p className="muted-copy">
          The timer includes queue waiting and stops while the audit is paused.
          Reaching the limit pauses new Runs; running work can finish.
        </p>
        {!valid ? (
          <p className="form-error" role="alert">
            Enter a duration greater than zero and no longer than 365 days.
          </p>
        ) : null}
        {error === null ? null : <AuditMutationNotice error={error} />}
        <div className="inline-actions">
          <button
            type="button"
            className="secondary-button"
            disabled={pending}
            onClick={onClose}
          >
            Close
          </button>
          <button type="submit" disabled={pending || !valid}>
            {pending ? "Applying…" : title}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
