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
  type ProjectKind,
} from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";

interface ProjectCollectionPresentation {
  kind: ProjectKind;
  heading: string;
  pageEyebrow: string;
  lede: string;
  createLabel: string;
  createHeading: string;
  collectionHeading: string;
  cardEyebrow: string;
  cardMark: string;
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
    queryKey: queryKeys.projects.list(presentation.kind, cursor),
    queryFn: () =>
      listProjects(api, {
        kind: presentation.kind,
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
      void navigate(
        `${presentation.detailRoot}/${encodeURIComponent(project.projectId)}`,
      );
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
        <button type="button" onClick={() => setCreateOpen((open) => !open)}>
          {createOpen ? "Close form" : presentation.createLabel}
        </button>
      </header>

      {createOpen ? (
        <form className="panel project-create-form" onSubmit={submit}>
          <div>
            <p className="eyebrow">Owner-scoped workspace</p>
            <h3>{presentation.createHeading}</h3>
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
            {create.isPending
              ? "Creating…"
              : `Create ${presentation.cardEyebrow}`}
          </button>
        </form>
      ) : null}

      <div className="project-list-heading">
        <div>
          <p className="eyebrow">Project kind</p>
          <h3>{presentation.collectionHeading}</h3>
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
          Loading {presentation.heading}…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="empty-state panel">
          <p className="eyebrow">Nothing here yet</p>
          <h3>{presentation.emptyHeading}</h3>
          <p>{presentation.emptyCopy}</p>
          <button type="button" onClick={() => setCreateOpen(true)}>
            {presentation.createLabel}
          </button>
        </div>
      ) : (
        <div className="project-card-grid">
          {query.data.items.map((project) => (
            <article
              className={`panel project-card ${project.lifecycle === "deleting" ? "is-deleting" : ""}`}
              key={project.projectId}
            >
              <div className="project-card-mark" aria-hidden="true">
                {presentation.cardMark}
              </div>
              <div>
                <p className="eyebrow project-card-eyebrow">
                  {presentation.cardEyebrow}
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
                to={`${presentation.detailRoot}/${encodeURIComponent(project.projectId)}`}
              >
                {project.lifecycle === "deleting"
                  ? "View deletion status →"
                  : `Open ${presentation.cardEyebrow} →`}
              </Link>
            </article>
          ))}
        </div>
      )}

      <CursorControls
        label={`${presentation.heading} pages`}
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

export function ProjectListRoute() {
  return (
    <ProjectCollectionRoute
      presentation={{
        kind: "project",
        heading: "Projects",
        pageEyebrow: "Reusable workspaces",
        lede: "Group reusable inputs, generated outputs, and related Workflow Runs without changing their exact Artifact provenance.",
        createLabel: "New Project",
        createHeading: "Create Project",
        collectionHeading: "Workspaces",
        cardEyebrow: "Project",
        cardMark: "P",
        detailRoot: "/projects",
        emptyHeading: "Create your first Project",
        emptyCopy:
          "Upload sources or other inputs once, then run multiple compatible Workflows against their exact revisions.",
      }}
    />
  );
}

export function EvaluationListRoute() {
  return (
    <ProjectCollectionRoute
      presentation={{
        kind: "evaluation",
        heading: "Evals",
        pageEyebrow: "Evaluation workspaces",
        lede: "Organize ordinary isolated Workflow Runs by eval metadata while retaining exact Project inputs and provenance.",
        createLabel: "New Eval",
        createHeading: "Create Eval workspace",
        collectionHeading: "Evaluation workspaces",
        cardEyebrow: "Eval",
        cardMark: "E",
        detailRoot: "/evals",
        emptyHeading: "Create your first Eval",
        emptyCopy:
          "An Eval is a Project workspace whose ordinary Runs use purpose=eval and eval.* metadata labels.",
      }}
    />
  );
}
