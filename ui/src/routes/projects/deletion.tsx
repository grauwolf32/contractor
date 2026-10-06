import { useId, useRef, useState } from "react";

import type { Project, ProjectDeletionPhase } from "../../api/projects";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { StatusGlyph } from "../../ui";

import "./projects.css";

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
    detail: "Remaining Project history is being removed.",
  },
};

/** The deletion phases in the order the Server moves through them. */
const DELETION_PHASES: readonly ProjectDeletionPhase[] = [
  "cancelling",
  "draining",
  "purging_runs",
  "purging_artifacts",
];

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
      <DialogHeader
        id={heading}
        eyebrow="Permanent workspace deletion"
        title={<>Delete {project.name}?</>}
      />
      <p className="project-delete-warning" id={warning}>
        This cancels every active Run and permanently deletes all Project Runs,
        execution history, and Project-scoped Artifacts. Shared User Artifacts,
        Skills, and Runtime credentials are retained.
      </p>
      <label className="projects-field">
        <span>
          Type <strong>{project.name}</strong> to confirm
        </span>
        <input
          value={confirmation}
          disabled={pending}
          autoComplete="off"
          onChange={(event) => setConfirmation(event.currentTarget.value)}
        />
      </label>
      {error === null ? null : <ErrorNotice error={error} />}
      <div className="projects-form-actions">
        <button
          ref={cancelButton}
          className="ui-btn"
          type="button"
          disabled={pending}
          onClick={onCancel}
        >
          Cancel
        </button>
        <button
          className="ui-btn"
          data-variant="danger"
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
  const heading = useId();
  const deletion = project.deletion;
  if (project.lifecycle !== "deleting" || deletion === undefined) {
    return null;
  }
  const copy = deletionPhaseCopy[deletion.phase];
  const current = DELETION_PHASES.indexOf(deletion.phase);
  return (
    <section
      className="projects-deletion"
      aria-labelledby={heading}
      aria-live="polite"
    >
      <p className="projects-deletion-state">
        <StatusGlyph tone="progress" />
        Deletion in progress
      </p>
      <h3 id={heading}>{copy.label}</h3>
      <p>{copy.detail}</p>
      <ol className="projects-deletion-phases" aria-label="Deletion phases">
        {DELETION_PHASES.map((phase, index) => {
          const state =
            index < current ? "done" : index === current ? "current" : "next";
          return (
            <li
              key={phase}
              data-state={state}
              aria-current={state === "current" ? "step" : undefined}
            >
              <StatusGlyph
                tone={
                  state === "done"
                    ? "done"
                    : state === "current"
                      ? "progress"
                      : "idle"
                }
              />
              <span>{deletionPhaseCopy[phase].label}</span>
              {state === "done" ? (
                <span className="ui-visually-hidden">, done</span>
              ) : null}
            </li>
          );
        })}
      </ol>
      <small>Requested {formatTimestamp(deletion.requestedAt)}</small>
      <p className="projects-deletion-durability">
        You can leave this page; cleanup resumes automatically after a Server
        restart.
      </p>
    </section>
  );
}
