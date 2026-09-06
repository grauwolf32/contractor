import { useId, useRef, useState } from "react";
import type { Project, ProjectDeletionPhase } from "../../api/projects";
import { Dialog } from "../../app/dialog";
import { ErrorNotice, formatTimestamp } from "../artifacts/common";

const deletionPhaseCopy: Record<
  ProjectDeletionPhase,
  { label: string; detail: string }
> = {
  cancelling: {
    label: "Cancelling active Runs",
    detail:
      "Every non-terminal Run is receiving the ordinary cancellation request.",
  },
  draining: {
    label: "Waiting for Runtime release",
    detail:
      "Cleanup is waiting for Runs to become terminal and allocations to be released.",
  },
  purging_runs: {
    label: "Removing Run history",
    detail:
      "Terminal Runs and their Run-owned Artifacts are being permanently removed.",
  },
  purging_artifacts: {
    label: "Removing Project Artifacts",
    detail:
      "The remaining ProjectScope history is being purged with reference-safe content cleanup.",
  },
};

export function DeleteProjectDialog({
  project,
  pending,
  error,
  onCancel,
  onConfirm,
}: {
  project: Project;
  pending: boolean;
  error: Error | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const warning = useId();
  const cancelButton = useRef<HTMLButtonElement>(null);
  const [confirmation, setConfirmation] = useState("");
  const resourceLabel = project.kind === "evaluation" ? "Eval" : "Project";
  return (
    <Dialog
      className="project-dialog project-delete-dialog panel"
      labelledBy={heading}
      describedBy={warning}
      initialFocusRef={cancelButton}
      onRequestClose={() => {
        if (!pending) onCancel();
      }}
      role="alertdialog"
    >
      <div className="project-dialog-heading">
        <div>
          <p className="eyebrow">Permanent workspace deletion</p>
          <h2 id={heading}>Delete {project.name}?</h2>
        </div>
      </div>
      <p className="project-delete-warning" id={warning}>
        This cancels every active Run and permanently deletes all Project Runs,
        execution history, and Project-scoped Artifacts. Shared User Artifacts,
        Skills, and Runtime credentials are retained.
      </p>
      <label>
        Type <strong>{project.name}</strong> to confirm
        <input
          value={confirmation}
          disabled={pending}
          autoComplete="off"
          onChange={(event) => setConfirmation(event.currentTarget.value)}
        />
      </label>
      {error === null ? null : <ErrorNotice error={error} />}
      <div className="run-delete-dialog-actions">
        <button
          ref={cancelButton}
          className="secondary-button"
          type="button"
          disabled={pending}
          onClick={onCancel}
        >
          Cancel
        </button>
        <button
          className="danger-button"
          type="button"
          disabled={pending || confirmation !== project.name}
          onClick={onConfirm}
        >
          {pending ? "Starting deletion…" : `Delete ${resourceLabel}`}
        </button>
      </div>
    </Dialog>
  );
}

export function ProjectDeletionProgress({ project }: { project: Project }) {
  const deletion = project.deletion;
  if (project.lifecycle !== "deleting" || deletion === undefined) {
    return null;
  }
  const copy = deletionPhaseCopy[deletion.phase];
  return (
    <div className="panel project-deletion-progress" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <div>
        <p className="eyebrow">Deletion in progress</p>
        <h3>{copy.label}</h3>
        <p>{copy.detail}</p>
        <small>Requested {formatTimestamp(deletion.requestedAt)}</small>
      </div>
      <p className="project-deletion-durability">
        You can leave this page. Cleanup is durable and resumes automatically
        after a Server restart.
      </p>
    </div>
  );
}
