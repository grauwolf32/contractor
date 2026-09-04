import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

import { usePublicAPI } from "../../api/context";
import {
  createProject,
  listProjects,
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  normalizeProjectRequest,
  type CreateProjectRequest,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";

export function ProjectListRoute() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const [createOpen, setCreateOpen] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const cursor = cursors.at(-1);
  const keyring = useMemo(
    () => new MutationDraftKeyring<CreateProjectRequest>("create-project"),
    [],
  );
  const query = useQuery({
    queryKey: queryKeys.projects.list("project", cursor),
    queryFn: () =>
      listProjects(api, {
        kind: "project",
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });
  const create = useMutation({
    mutationFn: (request: CreateProjectRequest) =>
      createProject(api, {
        request,
        idempotencyKey: keyring.keyFor(request),
      }),
    onSuccess: async (project) => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
      void navigate(`/projects/${encodeURIComponent(project.projectId)}`);
    },
  });

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    setValidationError(null);
    create.reset();
    const data = new FormData(event.currentTarget);
    try {
      create.mutate(
        normalizeProjectRequest({
          kind: "project",
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
    <section className="route-page projects-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Reusable workspaces</p>
          <h2>Projects</h2>
          <p className="lede">
            Group reusable inputs, generated outputs, and related Workflow Runs
            without changing their exact Artifact provenance.
          </p>
        </div>
        <button type="button" onClick={() => setCreateOpen((open) => !open)}>
          {createOpen ? "Close form" : "New Project"}
        </button>
      </header>

      {createOpen ? (
        <form className="panel project-create-form" onSubmit={submit}>
          <div>
            <p className="eyebrow">Owner-scoped workspace</p>
            <h3>Create Project</h3>
          </div>
          <div className="form-grid">
            <label>
              Name
              <input
                name="name"
                required
                maxLength={MAXIMUM_PROJECT_NAME_LENGTH}
                autoFocus
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
          {create.error === null ? null : <ErrorNotice error={create.error} />}
          <button type="submit" disabled={create.isPending}>
            {create.isPending ? "Creating…" : "Create Project"}
          </button>
        </form>
      ) : null}

      <div className="project-list-heading">
        <div>
          <p className="eyebrow">Project kind</p>
          <h3>Workspaces</h3>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading Projects…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="empty-state panel">
          <p className="eyebrow">Nothing here yet</p>
          <h3>Create your first Project</h3>
          <p>
            Upload sources or other inputs once, then run multiple compatible
            Workflows against their exact revisions.
          </p>
          <button type="button" onClick={() => setCreateOpen(true)}>
            New Project
          </button>
        </div>
      ) : (
        <div className="project-card-grid">
          {query.data.items.map((project) => (
            <article className="panel project-card" key={project.projectId}>
              <div className="project-card-mark" aria-hidden="true">
                P
              </div>
              <div>
                <p className="eyebrow">Project</p>
                <h3>
                  <Link
                    to={`/projects/${encodeURIComponent(project.projectId)}`}
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
              <dl className="project-card-meta">
                <div>
                  <dt>Updated</dt>
                  <dd>{formatTimestamp(project.updatedAt)}</dd>
                </div>
                <div>
                  <dt>ID</dt>
                  <dd>
                    <code>{project.projectId}</code>
                  </dd>
                </div>
              </dl>
              <Link
                className="project-card-open"
                to={`/projects/${encodeURIComponent(project.projectId)}`}
              >
                Open Project →
              </Link>
            </article>
          ))}
        </div>
      )}

      <CursorControls
        label="Project pages"
        canGoBack={cursors.length > 1}
        {...(query.data?.page.hasMore === true &&
        query.data.page.nextCursor !== undefined
          ? { nextCursor: query.data.page.nextCursor }
          : {})}
        onBack={() =>
          setCursors((current) =>
            current.slice(0, Math.max(1, current.length - 1)),
          )
        }
        onNext={(next) => setCursors((current) => [...current, next])}
      />
    </section>
  );
}
