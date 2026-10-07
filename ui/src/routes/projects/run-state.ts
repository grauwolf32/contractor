import type { WorkflowRunState } from "../../api/runs";
import type { VocabularyLabel } from "../../app/vocabulary";

/**
 * Words and tones for Workflow Run states on the project pages. Runs keep
 * their name (contract §3); the labels are the API states written as words,
 * so they never claim more than the state says.
 */
const RUN_STATE_LABELS: Readonly<Record<WorkflowRunState, VocabularyLabel>> = {
  initializing: { label: "Initializing", tone: "idle" },
  pending: { label: "Pending", tone: "idle" },
  waiting: { label: "Waiting", tone: "idle" },
  running: { label: "Running", tone: "progress" },
  cancelling: { label: "Cancelling", tone: "warning" },
  succeeded: { label: "Succeeded", tone: "done" },
  failed: { label: "Failed", tone: "blocked" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

/** "Succeeded", "Running", …; an unknown state reads as written. */
export function runStateLabel(state: WorkflowRunState): VocabularyLabel {
  return Object.hasOwn(RUN_STATE_LABELS, state)
    ? RUN_STATE_LABELS[state]
    : { label: String(state).replaceAll("_", " "), tone: "neutral" };
}
