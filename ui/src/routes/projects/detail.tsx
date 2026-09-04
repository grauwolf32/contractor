import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { Link, useParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import {
  getProject,
  listProjectRuns,
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  PROJECT_ID_PATTERN,
  updateProject,
  type Project,
} from "../../api/projects";
import type { RunSummary } from "../../api/runs";
import { queryKeys } from "../../api/query-keys";
import {
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "../runs/components";
import {
  ProjectArtifactDialog,
  ProjectArtifactShortcutGrid,
  ProjectRegion,
} from "./common";
import type { ShortcutDefinition } from "./shortcuts";
import { groupEvaluationRuns } from "./evaluation-groups";
import { ProjectWorkflowRecommendations } from "./workflow-recommendations";

function ProjectMetadataEditor({ project }: { project: Project }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: (request: { name: string; description: string }) =>
      updateProject(api, {
        projectId: project.projectId,
        expectedRevision: project.revision,
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
      setEditing(false);
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
          onClick={() => setEditing(true)}
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
            defaultValue={project.name}
          />
        </label>
        <label className="project-description-field">
          Description
          <textarea
            name="description"
            rows={3}
            maxLength={MAXIMUM_PROJECT_DESCRIPTION_LENGTH}
            defaultValue={project.description}
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
          onClick={() => setEditing(false)}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

function ProjectArtifactRegion({
  projectId,
  detailRoot,
}: {
  projectId: string;
  detailRoot: "/projects" | "/evals";
}) {
  const api = usePublicAPI();
  const [namespaceDraft, setNamespaceDraft] = useState("");
  const [namespace, setNamespace] = useState<string | undefined>();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const [shortcut, setShortcut] = useState<ShortcutDefinition | null>(null);
  const [written, setWritten] = useState<ArtifactWriteResponse | null>(null);
  const [filterError, setFilterError] = useState<string | null>(null);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.projects.artifacts.list(projectId, namespace, cursor),
    queryFn: () =>
      listProjectArtifacts(api, {
        projectId,
        ...(namespace === undefined ? {} : { namespace }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  function applyFilter(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const candidate = namespaceDraft.trim();
    if (candidate !== "" && !ARTIFACT_NAME_PATTERN.test(candidate)) {
      setFilterError("Namespace filter is not a valid Artifact name.");
      return;
    }
    setFilterError(null);
    setNamespace(candidate === "" ? undefined : candidate);
    setCursors([undefined]);
  }

  function finishUpload(result: ArtifactWriteResponse): void {
    setWritten(result);
    setShortcut(null);
    setNamespace(undefined);
    setNamespaceDraft("");
    setCursors([undefined]);
  }

  return (
    <ProjectRegion
      eyebrow="Reusable ProjectScope"
      title="Artifacts"
      id="project-artifacts"
      action={
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      }
    >
      <p className="muted-copy">
        Shortcuts suggest useful names and media types. Every field remains
        editable, and Other accepts any supported Artifact.
      </p>
      <ProjectArtifactShortcutGrid onSelect={setShortcut} />

      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Project Artifact revision stored.</strong>
          <Link
            to={`${detailRoot}/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(written.artifact.namespace)}/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open {written.artifact.namespace}/{written.artifact.name}@
            {written.artifact.revision}
          </Link>
        </div>
      )}

      <div className="project-artifact-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Current bindings</p>
            <h4>Artifact library</h4>
          </div>
          <form className="inline-form" onSubmit={applyFilter}>
            <label>
              Namespace
              <input
                name="namespaceFilter"
                placeholder="all namespaces"
                value={namespaceDraft}
                onChange={(event) => setNamespaceDraft(event.target.value)}
              />
            </label>
            <button className="secondary-button" type="submit">
              Apply
            </button>
          </form>
        </div>
        {filterError === null ? null : (
          <p className="form-error" role="alert">
            {filterError}
          </p>
        )}
        {query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading Project Artifacts…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No Artifact bindings in this view.</strong>
            <p>Use a shortcut above to add an exact Project input.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="responsive-table">
              <thead>
                <tr>
                  <th>Binding</th>
                  <th>Current revision</th>
                  <th>Media type</th>
                  <th>Size</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((item) => (
                  <tr key={`${item.artifact.namespace}/${item.artifact.name}`}>
                    <td data-label="Binding">
                      <Link
                        to={`${detailRoot}/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}`}
                      >
                        {item.artifact.namespace}/{item.artifact.name}
                      </Link>
                    </td>
                    <td data-label="Current revision">
                      <code>{item.artifact.revision}</code>
                    </td>
                    <td data-label="Media type">{item.mediaType}</td>
                    <td data-label="Size">{formatBytes(item.size)}</td>
                    <td data-label="Created">
                      {formatTimestamp(item.createdAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Project Artifact pages"
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
      </div>

      {shortcut === null ? null : (
        <ProjectArtifactDialog
          projectId={projectId}
          shortcut={shortcut}
          onClose={() => setShortcut(null)}
          onWritten={finishUpload}
        />
      )}
    </ProjectRegion>
  );
}

function EvaluationRuns({ runs }: { runs: readonly RunSummary[] }) {
  return (
    <div className="eval-run-groups">
      {groupEvaluationRuns(runs).map((group) => (
        <section className="eval-run-group" key={group.id}>
          <div className="eval-run-group-heading">
            <div>
              <p className="eyebrow">eval.id</p>
              <h4>
                {group.id === "" ? (
                  "Runs without eval.id"
                ) : (
                  <code>{group.id}</code>
                )}
              </h4>
            </div>
            <span className="project-workflow-count">
              {group.runs.length} {group.runs.length === 1 ? "Run" : "Runs"}
            </span>
          </div>
          {group.names.length === 0 ? null : (
            <p className="muted-copy">
              eval.name: <code>{group.names.join(", ")}</code>
            </p>
          )}
          <div className="table-scroll">
            <table className="responsive-table eval-run-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Leg</th>
                  <th>Case</th>
                  <th>Sample</th>
                  <th>Workflow</th>
                  <th>State</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {group.runs.map((run) => (
                  <tr key={run.runId}>
                    <td data-label="Run">
                      <Link to={`/runs/${encodeURIComponent(run.runId)}`}>
                        {run.runId}
                      </Link>
                    </td>
                    <td data-label="Leg">
                      <code>{run.labels["eval.leg"] ?? "—"}</code>
                    </td>
                    <td data-label="Case">
                      <span className="eval-case-value">
                        {run.labels["eval.fixture"] === undefined ? null : (
                          <small>{run.labels["eval.fixture"]}</small>
                        )}
                        <code>{run.labels["eval.case"] ?? "—"}</code>
                      </span>
                    </td>
                    <td data-label="Sample">
                      <code>{run.labels["eval.sample"] ?? "—"}</code>
                    </td>
                    <td data-label="Workflow">
                      <code>{run.workflow}</code>
                    </td>
                    <td data-label="State">
                      <StateBadge state={run.state} />
                    </td>
                    <td data-label="Updated">
                      {formatTimestamp(run.updatedAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}

function ProjectRunsRegion({
  projectId,
  evaluation,
}: {
  projectId: string;
  evaluation: boolean;
}) {
  const api = usePublicAPI();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.projects.runs(projectId, cursor),
    queryFn: () =>
      listProjectRuns(api, {
        projectId,
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  return (
    <ProjectRegion
      eyebrow="Execution history"
      title={evaluation ? "Eval Runs" : "Project Runs"}
      id="project-runs"
      action={<Link to="/runs">All Runs →</Link>}
    >
      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading {evaluation ? "Eval" : "Project"} Runs…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">
          <strong>
            No Workflow Runs belong to this {evaluation ? "Eval" : "Project"}.
          </strong>
          <p>Launch one compatible Workflow when inputs are ready.</p>
        </div>
      ) : evaluation ? (
        <EvaluationRuns runs={query.data.items} />
      ) : (
        <div className="table-scroll">
          <table className="responsive-table project-run-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Workflow</th>
                <th>State</th>
                <th>Labels</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((run) => (
                <tr key={run.runId}>
                  <td data-label="Run">
                    <Link to={`/runs/${encodeURIComponent(run.runId)}`}>
                      {run.runId}
                    </Link>
                  </td>
                  <td data-label="Workflow">
                    <code>{run.workflow}</code>
                  </td>
                  <td data-label="State">
                    <StateBadge state={run.state} />
                  </td>
                  <td data-label="Labels">
                    <RunMetadataLabelChips labels={run.labels} />
                  </td>
                  <td data-label="Updated">{formatTimestamp(run.updatedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorControls
        label="Project Run pages"
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
    </ProjectRegion>
  );
}

function ProjectWorkspaceRoute({
  expectedKind,
}: {
  expectedKind: "project" | "evaluation";
}) {
  const api = usePublicAPI();
  const { projectId = "" } = useParams();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validProject,
  });

  if (!validProject) {
    return (
      <section className="route-page">
        <ErrorNotice
          error={
            new Error(
              expectedKind === "evaluation"
                ? "Eval route is invalid"
                : "Project route is invalid",
            )
          }
        />
        <Link to={expectedKind === "evaluation" ? "/evals" : "/projects"}>
          Return to {expectedKind === "evaluation" ? "Evals" : "Projects"}
        </Link>
      </section>
    );
  }

  return (
    <section
      className={`route-page projects-page project-detail-page ${expectedKind === "evaluation" ? "eval-detail-page" : ""}`}
    >
      <header className="route-header-row">
        <div>
          <Link
            className="back-link"
            to={expectedKind === "evaluation" ? "/evals" : "/projects"}
          >
            ← All {expectedKind === "evaluation" ? "Evals" : "Projects"}
          </Link>
          <p className="eyebrow">
            {expectedKind === "evaluation"
              ? "Evaluation workspace"
              : "Project workspace"}
          </p>
          <h2>{project.data?.name ?? projectId}</h2>
          <p className="lede">
            {expectedKind === "evaluation"
              ? "Each sample remains an ordinary isolated Workflow Run; eval labels group it without changing execution semantics."
              : "Reusable inputs and published results stay Project-scoped; every Run still receives its own exact immutable copy."}
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={project.isFetching}
          onClick={() => void project.refetch()}
        >
          {project.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {project.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading Project…
        </p>
      ) : project.error !== null ? (
        <ErrorNotice error={project.error} />
      ) : project.data.kind !== expectedKind ? (
        <ErrorNotice
          error={
            new Error(
              expectedKind === "evaluation"
                ? "This workspace is available in Projects."
                : "Evaluation workspaces are available in Evals.",
            )
          }
        />
      ) : (
        <>
          <nav
            className="project-local-navigation"
            aria-label="Project sections"
          >
            <a href="#project-overview">Overview</a>
            <a href="#project-artifacts">Artifacts</a>
            <a href="#project-workflows">Workflows</a>
            <a href="#project-runs">Runs</a>
          </nav>
          <ProjectRegion
            eyebrow={
              expectedKind === "evaluation"
                ? "Evaluation metadata"
                : "Project metadata"
            }
            title="Overview"
            id="project-overview"
          >
            <ProjectMetadataEditor
              key={project.data.revision}
              project={project.data}
            />
          </ProjectRegion>
          <ProjectArtifactRegion
            projectId={project.data.projectId}
            detailRoot={expectedKind === "evaluation" ? "/evals" : "/projects"}
          />
          <ProjectWorkflowRecommendations projectId={project.data.projectId} />
          <ProjectRunsRegion
            projectId={project.data.projectId}
            evaluation={expectedKind === "evaluation"}
          />
        </>
      )}
    </section>
  );
}

export function ProjectDetailRoute() {
  return <ProjectWorkspaceRoute expectedKind="project" />;
}

export function EvaluationDetailRoute() {
  return <ProjectWorkspaceRoute expectedKind="evaluation" />;
}
