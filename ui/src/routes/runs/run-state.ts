/**
 * Display labels and tones for the Runs area. Runs keep the execution
 * vocabulary of S18 (contract §3: "Run, Workflow, Runtime Agent: unchanged"),
 * so only the case changes for display. Every table is keyed by a generated
 * API enum; a value this client does not know yet still gets a readable
 * label and the neutral tone.
 */
import type { QueueMembership } from "../../api/queue";
import type { StageAttempt, WorkflowRunState } from "../../api/runs";
import type { components } from "../../api/generated/public";
import type { StatusTone } from "../../app/status-tone";

export interface RunsLabel {
  readonly label: string;
  readonly tone: StatusTone;
}

type StageState = StageAttempt["state"];
type SubtaskStatus = components["schemas"]["PlannerSubtask"]["status"];
type AllocationStatus =
  components["schemas"]["StageRuntimeAllocation"]["status"];
type PublicationStatus = components["schemas"]["OutputPublication"]["status"];

export const RUN_STATE_LABELS: Readonly<Record<WorkflowRunState, RunsLabel>> = {
  initializing: { label: "Initializing", tone: "progress" },
  pending: { label: "Pending", tone: "idle" },
  waiting: { label: "Waiting", tone: "warning" },
  running: { label: "Running", tone: "progress" },
  cancelling: { label: "Cancelling", tone: "warning" },
  succeeded: { label: "Succeeded", tone: "done" },
  failed: { label: "Failed", tone: "blocked" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

export const MEMBERSHIP_LABELS: Readonly<Record<QueueMembership, string>> = {
  standalone: "Standalone",
  project: "Projects",
  evaluation: "Evaluations",
};

const STAGE_STATE_LABELS: Readonly<Record<StageState, RunsLabel>> = {
  preparing: { label: "Preparing", tone: "progress" },
  running: { label: "Running", tone: "progress" },
  finalizing: { label: "Finalizing", tone: "progress" },
  aborting: { label: "Aborting", tone: "warning" },
  succeeded: { label: "Succeeded", tone: "done" },
  failed: { label: "Failed", tone: "blocked" },
  interrupted: { label: "Interrupted", tone: "warning" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

const SUBTASK_STATUS_LABELS: Readonly<Record<SubtaskStatus, RunsLabel>> = {
  pending: { label: "Pending", tone: "idle" },
  running: { label: "Running", tone: "progress" },
  succeeded: { label: "Succeeded", tone: "done" },
  failed: { label: "Failed", tone: "blocked" },
};

const ALLOCATION_STATUS_LABELS: Readonly<Record<AllocationStatus, RunsLabel>> =
  {
    pinned: { label: "Pinned", tone: "progress" },
    release_pending: { label: "Release pending", tone: "warning" },
    released: { label: "Released", tone: "neutral" },
  };

const PUBLICATION_STATUS_LABELS: Readonly<
  Record<PublicationStatus, RunsLabel>
> = {
  published: { label: "Published", tone: "done" },
  already_present: { label: "Already present", tone: "neutral" },
  failed: { label: "Failed", tone: "blocked" },
};

/** "release_pending" → "Release pending", for values this client does not know. */
export function readableValue(value: string): string {
  const words = value.replaceAll(/[_-]+/g, " ").trim();
  return words === ""
    ? "Unknown"
    : words.charAt(0).toUpperCase() + words.slice(1);
}

function lookup<K extends string>(
  table: Readonly<Record<K, RunsLabel>>,
  value: K,
): RunsLabel {
  return Object.hasOwn(table, value)
    ? table[value]
    : { label: readableValue(value), tone: "neutral" };
}

/** WorkflowRun state: "Running", "Succeeded", "Failed", … */
export function runStateLabel(state: WorkflowRunState): RunsLabel {
  return lookup(RUN_STATE_LABELS, state);
}

/** StageExecution (attempt) state. */
export function stageStateLabel(state: StageState): RunsLabel {
  return lookup(STAGE_STATE_LABELS, state);
}

/** Planner subtask status. */
export function subtaskStatusLabel(status: SubtaskStatus): RunsLabel {
  return lookup(SUBTASK_STATUS_LABELS, status);
}

/** Allocation-pinned Runtime configuration status. */
export function allocationStatusLabel(status: AllocationStatus): RunsLabel {
  return lookup(ALLOCATION_STATUS_LABELS, status);
}

/** Project output publication receipt status. */
export function publicationStatusLabel(status: PublicationStatus): RunsLabel {
  return lookup(PUBLICATION_STATUS_LABELS, status);
}
