import { useMutation, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useId, useRef, useState } from "react";

import { usePublicAPI } from "../../api/context";
import { invalidateCrossProject } from "../../api/cross-project";
import {
  createProject,
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  normalizeProjectRequest,
  type CreateProjectRequest,
  type Project,
  type ProjectKind,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { Dialog, DialogHeader } from "../../app/dialog";
import { ErrorNotice } from "../../app/error-notice";
import type { NewProjectKeys } from "./new-project-keys";

import "./projects.css";

export interface NewProjectWording {
  /** Dialog title, e.g. "New project". */
  title: string;
  /** Accessible name of the close button. */
  closeLabel: string;
  /** Submit button, e.g. "Create project". */
  submitLabel: string;
}

/**
 * Creates a project or an evaluation workspace: name (≤ 160) and
 * description (≤ 4096). An identical request reuses its Idempotency-Key,
 * also after the dialog was closed and opened again (the opening page keeps
 * `keys`), so a retry after a lost response cannot create a second project.
 */
export function NewProjectDialog({
  kind,
  keys,
  wording,
  onClose,
  onCreated,
}: {
  kind: ProjectKind;
  /** From useNewProjectKeys() in the page that opens the dialog. */
  keys: NewProjectKeys;
  wording: NewProjectWording;
  onClose: () => void;
  onCreated: (project: Project) => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const heading = useId();
  const nameField = useRef<HTMLInputElement>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const create = useMutation({
    mutationFn: (request: CreateProjectRequest) =>
      createProject(api, {
        request,
        idempotencyKey: keys.keyFor(request),
      }),
    onSuccess: async (project) => {
      keys.created();
      // Lists elsewhere refresh in the background; the dialog waits only for
      // the list the new project joins.
      void invalidateCrossProject(queryClient);
      if (kind === "evaluation")
        void queryClient.invalidateQueries({
          queryKey: queryKeys.evals.projects,
        });
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.lists(kind),
      });
      onCreated(project);
    },
  });

  function close(): void {
    if (create.isPending) return;
    onClose();
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    create.reset();
    const data = new FormData(event.currentTarget);
    try {
      create.mutate(
        normalizeProjectRequest({
          kind,
          name: String(data.get("name") ?? ""),
          description: String(data.get("description") ?? ""),
        }),
      );
    } catch (error) {
      setValidationError(
        error instanceof Error ? error.message : "Project metadata is invalid",
      );
    }
  }

  return (
    <Dialog
      className="project-dialog panel projects-new-dialog"
      labelledBy={heading}
      initialFocusRef={nameField}
      onRequestClose={close}
    >
      <DialogHeader
        id={heading}
        title={wording.title}
        close={{
          label: wording.closeLabel,
          disabled: create.isPending,
          onClose: close,
        }}
      />
      <form className="projects-form" onSubmit={submit}>
        <label className="projects-field">
          <span>Name</span>
          <input
            ref={nameField}
            name="name"
            required
            autoComplete="off"
            maxLength={MAXIMUM_PROJECT_NAME_LENGTH}
          />
        </label>
        <label className="projects-field">
          <span>Description</span>
          <textarea
            name="description"
            maxLength={MAXIMUM_PROJECT_DESCRIPTION_LENGTH}
            rows={3}
          />
        </label>
        {validationError === null ? null : (
          <p className="form-error" role="alert">
            {validationError}
          </p>
        )}
        {create.error === null ? null : <ErrorNotice error={create.error} />}
        <div className="projects-form-actions">
          <button
            type="button"
            className="ui-btn"
            disabled={create.isPending}
            onClick={close}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="ui-btn"
            data-variant="primary"
            disabled={create.isPending}
          >
            {create.isPending ? "Creating…" : wording.submitLabel}
          </button>
        </div>
      </form>
    </Dialog>
  );
}
