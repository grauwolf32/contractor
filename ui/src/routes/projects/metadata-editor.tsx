import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { invalidateCrossProject } from "../../api/cross-project";
import {
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  updateProject,
  type Project,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { IdChip } from "../../ui";

import "./projects.css";

/** Kind, revision and record times: internals for Technical details. */
export function ProjectFacts({
  project,
  showId = true,
}: {
  project: Project;
  /** Off where the page header already shows the project ID. */
  showId?: boolean;
}) {
  return (
    <dl className="projects-facts">
      {showId ? (
        <div>
          <dt>Project ID</dt>
          <dd>
            <IdChip value={project.projectId} label="project ID" />
          </dd>
        </div>
      ) : null}
      <div>
        <dt>Kind</dt>
        <dd>{project.kind}</dd>
      </div>
      <div>
        <dt>Revision</dt>
        <dd>
          <code>{project.revision}</code>
        </dd>
      </div>
      <div>
        <dt>Created</dt>
        <dd>{formatTimestamp(project.createdAt)}</dd>
      </div>
      <div>
        <dt>Updated</dt>
        <dd>{formatTimestamp(project.updatedAt)}</dd>
      </div>
    </dl>
  );
}

/**
 * Name and description. Editing keeps the revision it started from: a save
 * sends it as If-Match, so a change made elsewhere meanwhile is refused (412)
 * and explained instead of overwritten; the draft stays for a new attempt.
 */
export function ProjectMetadataEditor({ project }: { project: Project }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState<Project | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: (request: { name: string; description: string }) =>
      updateProject(api, {
        projectId: project.projectId,
        expectedRevision: editing?.revision ?? project.revision,
        request,
      }),
    onSuccess: async (updated) => {
      queryClient.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.projects.lists(project.kind),
        }),
        // Cross-project lists show the project's name.
        invalidateCrossProject(queryClient),
      ]);
      setEditing(null);
    },
  });

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    mutation.reset();
    const data = new FormData(event.currentTarget);
    const name = String(data.get("name") ?? "").trim();
    const description = String(data.get("description") ?? "").trim();
    if (
      name.length === 0 ||
      name.length > MAXIMUM_PROJECT_NAME_LENGTH ||
      description.length > MAXIMUM_PROJECT_DESCRIPTION_LENGTH
    ) {
      setValidationError("Project metadata is invalid.");
      return;
    }
    mutation.mutate({ name, description });
  }

  if (!editing) {
    return (
      <div className="projects-editor">
        <dl className="projects-facts">
          <div>
            <dt>Name</dt>
            <dd>{project.name}</dd>
          </div>
          <div className="projects-facts-wide">
            <dt>Description</dt>
            <dd className="projects-prose">
              {project.description === ""
                ? "No description provided."
                : project.description}
            </dd>
          </div>
        </dl>
        <div className="projects-form-actions">
          <button
            className="ui-btn"
            data-size="sm"
            type="button"
            onClick={() => {
              mutation.reset();
              setValidationError(null);
              setEditing(project);
            }}
          >
            Edit metadata
          </button>
        </div>
      </div>
    );
  }

  return (
    <form className="projects-form" onSubmit={submit}>
      <label className="projects-field">
        <span>Name</span>
        <input
          name="name"
          required
          autoComplete="off"
          maxLength={MAXIMUM_PROJECT_NAME_LENGTH}
          defaultValue={editing.name}
        />
      </label>
      <label className="projects-field">
        <span>Description</span>
        <textarea
          name="description"
          rows={3}
          maxLength={MAXIMUM_PROJECT_DESCRIPTION_LENGTH}
          defaultValue={editing.description}
        />
      </label>
      {validationError === null ? null : (
        <p className="form-error" role="alert">
          {validationError}
        </p>
      )}
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <div className="projects-form-actions">
        <button
          className="ui-btn"
          data-variant="primary"
          data-size="sm"
          type="submit"
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Saving…" : "Save changes"}
        </button>
        <button
          className="ui-btn"
          data-size="sm"
          type="button"
          disabled={mutation.isPending}
          onClick={() => setEditing(null)}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}
