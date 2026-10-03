import { useDocumentTitle } from "../../app/document-title";
import { RecordedTime } from "../../app/recorded-time";
import { EvaluationActivity } from "./evaluation-activity";
import "./collection.css";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useId, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router";

import { usePublicAPI } from "../../api/context";
import {
  createProject,
  deleteProject,
  listProjects,
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  normalizeProjectRequest,
  type CreateProjectRequest,
  type Project,
  type ProjectKind,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { DeleteIcon } from "../../app/delete-icon";
import { Dialog, DialogHeader } from "../../app/dialog";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import { CursorControls } from "../../app/cursor-controls";
import { useCursorStack } from "../../app/pagination";
import { ErrorNotice } from "../../app/error-notice";
import { DeleteProjectDialog } from "./deletion";
import { RefreshButton } from "../../app/refresh-button";
import { QueryView } from "../../app/query-view";
import { compactId } from "../../app/format";

interface ProjectCollectionPresentation {
  kind: ProjectKind;
  heading: string;
  pageEyebrow: string;
  lede: string;
  createLabel: string;
  collectionHeading: string;
  cardEyebrow: string;
  detailRoot: "/projects" | "/evals";
  emptyHeading: string;
  emptyCopy: string;
}

function ProjectCollectionRoute({
  presentation,
}: {
  presentation: ProjectCollectionPresentation;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const pages = useCursorStack();
  const [createOpen, setCreateOpen] = useState(false);
  const createHeading = useId();
  const createNameField = useRef<HTMLInputElement>(null);
  const [deleteTarget, setDeleteTarget] = useState<Project>();
  const [validationError, setValidationError] = useState<string | null>(null);
  const cursor = pages.cursor;
  const keyring = useMemo(
    () => new MutationDraftKeyring<CreateProjectRequest>("create-project"),
    [],
  );
  const query = useQuery({
    queryKey: queryKeys.projects.list(presentation.kind, cursor),
    queryFn: () =>
      listProjects(api, {
        kind: presentation.kind,
        ...(cursor === undefined ? {} : { cursor }),
      }),
    refetchInterval: (query) =>
      query.state.data?.items.some(
        (project) => project.lifecycle === "deleting",
      )
        ? 1_000
        : false,
  });
  const create = useMutation({
    mutationFn: (request: CreateProjectRequest) =>
      createProject(api, {
        request,
        idempotencyKey: keyring.keyFor(request),
      }),
    onSuccess: async (project) => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
      void navigate(
        `${presentation.detailRoot}/${encodeURIComponent(project.projectId)}`,
      );
    },
  });
  const deletion = useMutation({
    mutationFn: (project: Project) =>
      deleteProject(api, {
        projectId: project.projectId,
        expectedRevision: project.revision,
      }),
    onSuccess: async (deleting) => {
      setDeleteTarget(undefined);
      queryClient.setQueryData(
        queryKeys.projects.detail(deleting.projectId),
        deleting,
      );
      await queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
    },
    onError: () =>
      queryClient.invalidateQueries({ queryKey: queryKeys.projects.all }),
  });

  function closeCreate(): void {
    if (create.isPending) return;
    setCreateOpen(false);
    setValidationError(null);
    create.reset();
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    create.reset();
    const data = new FormData(event.currentTarget);
    try {
      create.mutate(
        normalizeProjectRequest({
          kind: presentation.kind,
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
    <section
      className={`route-page projects-page ${presentation.kind === "evaluation" ? "evals-page" : ""}`}
    >
      <header className="route-header-row">
        <div>
          <p className="eyebrow">{presentation.pageEyebrow}</p>
          <h2>{presentation.heading}</h2>
          <p className="lede">{presentation.lede}</p>
        </div>
        <button type="button" onClick={() => setCreateOpen(true)}>
          {presentation.createLabel}
        </button>
      </header>

      {createOpen ? (
        <Dialog
          className="project-dialog panel"
          labelledBy={createHeading}
          initialFocusRef={createNameField}
          onRequestClose={closeCreate}
        >
          <DialogHeader
            id={createHeading}
            eyebrow="Workspace"
            title={presentation.createLabel}
            close={{
              label: `Close ${presentation.createLabel} form`,
              disabled: create.isPending,
              onClose: closeCreate,
            }}
          />
          <form className="project-dialog-form" onSubmit={submit}>
            <div className="form-grid">
              <label>
                Name
                <input
                  ref={createNameField}
                  name="name"
                  required
                  maxLength={MAXIMUM_PROJECT_NAME_LENGTH}
                />
              </label>
              <label className="project-description-field">
                Description
                <textarea
                  name="description"
                  maxLength={MAXIMUM_PROJECT_DESCRIPTION_LENGTH}
                  rows={3}
                />
              </label>
            </div>
            {validationError === null ? null : (
              <p className="form-error" role="alert">
                {validationError}
              </p>
            )}
            {create.error === null ? null : (
              <ErrorNotice error={create.error} />
            )}
            <div className="project-dialog-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={create.isPending}
                onClick={closeCreate}
              >
                Cancel
              </button>
              <button type="submit" disabled={create.isPending}>
                {create.isPending
                  ? "Creating…"
                  : `Create ${presentation.cardEyebrow}`}
              </button>
            </div>
          </form>
        </Dialog>
      ) : null}

      <div className="project-list-heading">
        <div>
          <p className="eyebrow">Project kind</p>
          <h3>{presentation.collectionHeading}</h3>
        </div>
        <RefreshButton
          isFetching={query.isFetching}
          onRefresh={() => void query.refetch()}
          label="Refresh"
        />
      </div>

      <QueryView
        query={query}
        loading={
          <p className="loading-copy" role="status">
            Loading {presentation.heading}…
          </p>
        }
        errorContext={`Could not load ${presentation.kind === "evaluation" ? "Eval workspaces" : "Projects"}`}
        onRetry={() => void query.refetch()}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <div className="empty-state panel">
            <p className="eyebrow">Nothing here yet</p>
            <h3>{presentation.emptyHeading}</h3>
            <p>{presentation.emptyCopy}</p>
            <button type="button" onClick={() => setCreateOpen(true)}>
              {presentation.createLabel}
            </button>
          </div>
        }
      >
        {(data) => (
          <div className="project-card-grid">
            {data.items.map((project) => (
              <article
                className={`panel project-card ${project.lifecycle === "deleting" ? "is-deleting" : ""}`}
                key={project.projectId}
              >
                <div>
                  <p className="eyebrow project-card-eyebrow">
                    {project.lifecycle === "deleting" ? (
                      <span>Deletion in progress</span>
                    ) : null}
                  </p>
                  <h3>
                    <Link
                      to={`${presentation.detailRoot}/${encodeURIComponent(project.projectId)}`}
                    >
                      {project.name}
                    </Link>
                  </h3>
                  <p className="project-card-description">
                    {project.description === ""
                      ? "No description provided."
                      : project.description}
                  </p>
                </div>
                {presentation.kind === "evaluation" ? (
                  <EvaluationActivity projectId={project.projectId} />
                ) : null}
                <dl className="project-card-meta">
                  <div>
                    <dt>Updated</dt>
                    <dd>
                      <RecordedTime value={project.updatedAt} />
                    </dd>
                  </div>
                  <div>
                    <dt>ID</dt>
                    <dd>
                      <code title={project.projectId}>
                        {compactId(project.projectId)}
                      </code>
                    </dd>
                  </div>
                </dl>
                <div className="project-card-actions">
                  <Link
                    className="project-card-open"
                    to={`${presentation.detailRoot}/${encodeURIComponent(project.projectId)}`}
                  >
                    {project.lifecycle === "deleting"
                      ? "View deletion status →"
                      : `Open ${presentation.cardEyebrow} →`}
                  </Link>
                  {presentation.kind === "project" ? (
                    <button
                      className="danger-button delete-icon-button"
                      type="button"
                      aria-label={`Delete Project ${project.name}`}
                      title={
                        project.lifecycle === "deleting"
                          ? "Project deletion in progress"
                          : "Delete Project"
                      }
                      disabled={
                        project.lifecycle === "deleting" || deletion.isPending
                      }
                      onClick={() => {
                        deletion.reset();
                        setDeleteTarget(project);
                      }}
                    >
                      <DeleteIcon />
                    </button>
                  ) : null}
                </div>
              </article>
            ))}
          </div>
        )}
      </QueryView>

      <CursorControls
        label={`${presentation.heading} pages`}
        {...pages.controls(query.data?.page)}
      />
      {deleteTarget === undefined ? null : (
        <DeleteProjectDialog
          key={deleteTarget.projectId}
          project={deleteTarget}
          pending={deletion.isPending}
          error={deletion.error}
          onCancel={() => {
            if (deletion.isPending) return;
            setDeleteTarget(undefined);
            deletion.reset();
          }}
          onConfirm={() => {
            if (!deletion.isPending) deletion.mutate(deleteTarget);
          }}
        />
      )}
    </section>
  );
}

export function ProjectListRoute() {
  useDocumentTitle("Projects");
  return (
    <ProjectCollectionRoute
      presentation={{
        kind: "project",
        heading: "Projects",
        pageEyebrow: "Reusable workspaces",
        lede: "Keep source materials, analysis results and related Runs in one workspace.",
        createLabel: "New Project",
        collectionHeading: "Workspaces",
        cardEyebrow: "Project",
        detailRoot: "/projects",
        emptyHeading: "Create your first Project",
        emptyCopy:
          "Upload sources or other inputs once, then run multiple compatible Workflows against them.",
      }}
    />
  );
}

export function EvaluationListRoute() {
  useDocumentTitle("Evals");
  return (
    <ProjectCollectionRoute
      presentation={{
        kind: "evaluation",
        heading: "Evals",
        pageEyebrow: "Evaluation workspaces",
        lede: "Compare execution history across evaluation workspaces. A succeeded Run is an execution status; review outputs for the evaluation result.",
        createLabel: "New Eval",
        collectionHeading: "Evaluation workspaces",
        cardEyebrow: "Eval",
        detailRoot: "/evals",
        emptyHeading: "Create your first Eval",
        emptyCopy:
          "An Eval is a Project workspace whose ordinary Runs use purpose=eval and eval.* metadata labels.",
      }}
    />
  );
}
