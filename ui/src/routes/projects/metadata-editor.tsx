import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { usePublicAPI } from "../../api/context";
import {
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  updateProject,
  type Project,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice, formatTimestamp } from "../artifacts/common";

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
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.list(project.kind),
      });
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
      <>
        <dl className="metadata-grid project-metadata-grid">
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
          <div className="project-description-value">
            <dt>Description</dt>
            <dd>
              {project.description === ""
                ? "No description provided."
                : project.description}
            </dd>
          </div>
        </dl>
        <button
          className="secondary-button"
          type="button"
          onClick={() => setEditing(project)}
        >
          Edit metadata
        </button>
      </>
    );
  }

  return (
    <form className="project-metadata-form" onSubmit={submit}>
      <div className="form-grid">
        <label>
          Name
          <input
            name="name"
            required
            maxLength={MAXIMUM_PROJECT_NAME_LENGTH}
            defaultValue={editing.name}
          />
        </label>
        <label className="project-description-field">
          Description
          <textarea
            name="description"
            rows={3}
            maxLength={MAXIMUM_PROJECT_DESCRIPTION_LENGTH}
            defaultValue={editing.description}
          />
        </label>
      </div>
      {validationError === null ? null : (
        <p className="form-error" role="alert">
          {validationError}
        </p>
      )}
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <div className="project-form-actions">
        <button type="submit" disabled={mutation.isPending}>
          {mutation.isPending ? "Saving…" : "Save exact update"}
        </button>
        <button
          className="secondary-button"
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
