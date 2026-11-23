import { writeConflict, type WriteConflict } from "./conflict";

/**
 * What a refused write tells the user. `subject` starts the sentence: "The
 * check", "These settings", "This credential".
 */
function conflictMessage(conflict: WriteConflict, subject: string): string {
  return conflict === "stale"
    ? `${subject} changed in the meantime. The page now shows its current state; review it before trying again.`
    : `${subject} is no longer in a state that allows this, so nothing was applied. The page now shows its current state.`;
}

/**
 * The write-conflict sentence under a form or confirmation, as a polite
 * status. Renders nothing for errors that are no write conflict, so it can
 * sit next to an ErrorNotice that shows the Server's own message.
 */
export function ConflictNotice({
  error,
  subject,
  className = "ui-caption",
}: {
  error: unknown;
  subject: string;
  className?: string;
}) {
  const conflict = writeConflict(error);
  if (conflict === undefined) return null;
  return (
    <p className={className} role="status">
      {conflictMessage(conflict, subject)}
    </p>
  );
}
