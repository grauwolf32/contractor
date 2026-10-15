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
import { Button, Facts, IdChip } from "../../ui";
import { useUnsavedDraft } from "./project-drafts";

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
    <Facts
      size="sm"
      items={[
        {
          term: "Project ID",
          value: showId && (
            <IdChip value={project.projectId} label="project ID" />
          ),
        },
        { term: "Kind", value: project.kind },
        { term: "Revision", value: <code>{project.revision}</code> },
        { term: "Created", value: formatTimestamp(project.createdAt) },
        { term: "Updated", value: formatTimestamp(project.updatedAt) },
      ]}
    />
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
  useUnsavedDraft(editing !== null);
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
      // Cross-project lists show the project's name: they refresh in the
      // background, the save waits only for the project lists.
      void invalidateCrossProject(queryClient);
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.lists(project.kind),
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
        <div className="ui-toolbar">
          <Button
            size="sm"
            onClick={() => {
              mutation.reset();
              setValidationError(null);
              setEditing(project);
            }}
          >
            Edit metadata
          </Button>
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
      <div className="ui-toolbar">
        <Button
          variant="primary"
          size="sm"
          type="submit"
          pending={mutation.isPending}
          pendingLabel="Saving…"
        >
          Save changes
        </Button>
        <Button
          size="sm"
          disabled={mutation.isPending}
          onClick={() => setEditing(null)}
        >
          Cancel
        </Button>
      </div>
    </form>
  );
}
