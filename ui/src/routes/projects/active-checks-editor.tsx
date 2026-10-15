import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import { usePublicAPI } from "../../api/context";
import {
  requireActiveChecksPolicy,
  updateProject,
  type ActiveChecksPolicy,
  type Project,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import { useUnsavedDraft } from "./project-drafts";

const MODES: Record<ActiveChecksPolicy, string> = {
  automatic: "Automatically",
  "approval-required": "Ask before each active check",
  prohibited: "Disabled",
};

export function ProjectActiveChecksEditor({ project }: { project: Project }) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const [editing, setEditing] = useState<Project | null>(null);
  useUnsavedDraft(editing !== null);
  const mutation = useMutation({
    mutationFn: (activeChecks: ActiveChecksPolicy) =>
      updateProject(api, {
        projectId: project.projectId,
        expectedRevision: editing?.revision ?? project.revision,
        request: { activeChecks },
      }),
    onSuccess: (updated) => {
      client.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      void client.invalidateQueries({
        queryKey: queryKeys.projects.lists(project.kind),
      });
      setEditing(null);
    },
  });

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    mutation.mutate(
      requireActiveChecksPolicy(
        new FormData(event.currentTarget).get("activeChecks"),
      ),
    );
  }

  if (editing === null)
    return (
      <div className="projects-editor">
        <p>{MODES[project.activeChecks]}</p>
        <p className="projects-caption">
          Applies when starting a new check. Checks already in progress keep
          their setting. Presets that disable active tests remain disabled.
        </p>
        <button
          className="ui-btn"
          data-size="sm"
          type="button"
          onClick={() => {
            mutation.reset();
            setEditing(project);
          }}
        >
          Edit active checks
        </button>
      </div>
    );

  return (
    <form className="projects-form" onSubmit={submit}>
      <label className="projects-field">
        <span>Active checks</span>
        <select name="activeChecks" defaultValue={editing.activeChecks}>
          {Object.entries(MODES).map(([mode, label]) => (
            <option key={mode} value={mode}>
              {label}
            </option>
          ))}
        </select>
      </label>
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <div className="projects-form-actions">
        <button
          className="ui-btn"
          data-size="sm"
          data-variant="primary"
          type="submit"
          disabled={mutation.isPending}
        >
          {mutation.isPending ? "Saving…" : "Save active checks"}
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
