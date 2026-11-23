import { useId, useState } from "react";

import type { Project, ProjectDeletionPhase } from "../../api/projects";
import { ActionMenu } from "../../app/action-menu";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { formatTimestamp } from "../../app/format";
import { AlertDialog, Button, Field, IconButton, StatusGlyph } from "../../ui";
import type { WorkspaceProject } from "./use-project-deletion";

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

/**
 * Explicit Project header actions; the Eval Workspace keeps its own menu.
 * Both use the existing controller and named deletion confirmation.
 */
export function ProjectActions({
  workspace,
  noun,
  onRefresh,
  refreshing,
}: {
  workspace: WorkspaceProject;
  noun: "Project" | "Workspace";
  onRefresh?: (() => void) | undefined;
  refreshing?: boolean | undefined;
}) {
  const { project } = workspace;
  if (noun === "Project")
    return (
      <div className="projects-header-actions">
        <IconButton
          label="Refresh project details"
          size="sm"
          icon={<Icon name="refresh" />}
          pending={refreshing ?? project.isFetching}
          onClick={onRefresh ?? (() => void project.refetch())}
        />
        {workspace.activeProject === undefined ? null : (
          <Button
            variant="ghost"
            size="sm"
            className="projects-delete"
            icon={<Icon name="delete" />}
            onClick={workspace.requestDeletion}
          >
            Delete Project
          </Button>
        )}
      </div>
    );
  return (
    <ActionMenu label={`${noun} actions`}>
      <Button
        variant="ghost"
        size="sm"
        pending={project.isFetching}
        pendingLabel="Refreshing…"
        onClick={() => void project.refetch()}
      >
        Refresh project details
      </Button>
      {workspace.activeProject === undefined ? null : (
        <Button variant="danger" size="sm" onClick={workspace.requestDeletion}>
          Delete {noun}
        </Button>
      )}
    </ActionMenu>
  );
}

/** The type-to-confirm deletion dialog, while the workspace asks for it. */
export function DeleteProjectDialog({
  workspace,
}: {
  workspace: WorkspaceProject;
}) {
  const project = workspace.activeProject;
  if (!workspace.deleteOpen || project === undefined) return null;
  return <DeleteConfirmation project={project} workspace={workspace} />;
}

function DeleteConfirmation({
  project,
  workspace: { deletion, cancelDeletion },
}: {
  project: Project;
  workspace: WorkspaceProject;
}) {
  const [confirmation, setConfirmation] = useState("");
  const pending = deletion.isPending;
  const resourceLabel = project.kind === "evaluation" ? "Workspace" : "Project";
  return (
    <AlertDialog
      eyebrow="Permanent workspace deletion"
      title={<>Delete {project.name}?</>}
      confirmLabel={`Delete ${resourceLabel}`}
      pendingLabel="Starting deletion…"
      pending={pending}
      confirmDisabled={confirmation !== project.name}
      details={
        <Field
          label={
            <>
              Type <strong>{project.name}</strong> to confirm
            </>
          }
        >
          {(control) => (
            <input
              {...control}
              value={confirmation}
              disabled={pending}
              autoComplete="off"
              onChange={(event) => setConfirmation(event.currentTarget.value)}
            />
          )}
        </Field>
      }
      error={
        deletion.error === null ? null : <ErrorNotice error={deletion.error} />
      }
      onCancel={cancelDeletion}
      onConfirm={() => deletion.mutate(project)}
    >
      <p>
        This cancels every active Run and permanently deletes all Project Runs,
        execution history, and Project-scoped Artifacts. Shared User Artifacts,
        Skills, and Runtime credentials are retained.
      </p>
    </AlertDialog>
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
