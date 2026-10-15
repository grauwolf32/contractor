/**
 * Display labels and tones of Run parts: stage attempts, Planner subtasks,
 * pinned configurations and output publications. Run states themselves are
 * in src/app/vocabulary.ts. Runs keep the execution vocabulary of S18
 * (contract §3: "Run, Workflow, Runtime Agent: unchanged"), so only the case
 * changes for display. Every table is keyed by a generated API enum; a value
 * this client does not know yet still gets a readable label and the neutral
 * tone.
 */
import type { QueueMembership } from "../../api/queue";
import type { StageAttempt } from "../../api/runs";
import type { components } from "../../api/generated/public";
import { labelOf, type VocabularyLabel } from "../../app/vocabulary";

type StageState = StageAttempt["state"];
type SubtaskStatus = components["schemas"]["PlannerSubtask"]["status"];
type AllocationStatus =
  components["schemas"]["StageRuntimeAllocation"]["status"];
type PublicationStatus = components["schemas"]["OutputPublication"]["status"];

export const MEMBERSHIP_LABELS: Readonly<Record<QueueMembership, string>> = {
  standalone: "Standalone",
  project: "Projects",
  evaluation: "Evaluations",
};

const STAGE_STATE_LABELS: Readonly<Record<StageState, VocabularyLabel>> = {
  preparing: { label: "Preparing", tone: "progress" },
  running: { label: "Running", tone: "progress" },
  finalizing: { label: "Finalizing", tone: "progress" },
  aborting: { label: "Aborting", tone: "warning" },
  succeeded: { label: "Succeeded", tone: "done" },
  failed: { label: "Failed", tone: "blocked" },
  interrupted: { label: "Interrupted", tone: "warning" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

const SUBTASK_STATUS_LABELS: Readonly<Record<SubtaskStatus, VocabularyLabel>> =
  {
    pending: { label: "Pending", tone: "idle" },
    running: { label: "Running", tone: "progress" },
    succeeded: { label: "Succeeded", tone: "done" },
    failed: { label: "Failed", tone: "blocked" },
  };

const ALLOCATION_STATUS_LABELS: Readonly<
  Record<AllocationStatus, VocabularyLabel>
> = {
  pinned: { label: "Pinned", tone: "progress" },
  release_pending: { label: "Release pending", tone: "warning" },
  released: { label: "Released", tone: "neutral" },
};

const PUBLICATION_STATUS_LABELS: Readonly<
  Record<PublicationStatus, VocabularyLabel>
> = {
  published: { label: "Published", tone: "done" },
  already_present: { label: "Already present", tone: "neutral" },
  failed: { label: "Failed", tone: "blocked" },
};

/** StageExecution (attempt) state. */
export function stageStateLabel(state: StageState): VocabularyLabel {
  return labelOf(STAGE_STATE_LABELS, state);
}

/** Planner subtask status. */
export function subtaskStatusLabel(status: SubtaskStatus): VocabularyLabel {
  return labelOf(SUBTASK_STATUS_LABELS, status);
}

/** Allocation-pinned Runtime configuration status. */
export function allocationStatusLabel(
  status: AllocationStatus,
): VocabularyLabel {
  return labelOf(ALLOCATION_STATUS_LABELS, status);
}

/** Project output publication receipt status. */
export function publicationStatusLabel(
  status: PublicationStatus,
): VocabularyLabel {
  return labelOf(PUBLICATION_STATUS_LABELS, status);
}
