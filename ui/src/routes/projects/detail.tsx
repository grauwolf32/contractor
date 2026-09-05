import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useId, useState } from "react";
import { Link, useNavigate, useParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { listProjectArtifacts } from "../../api/project-artifacts";
import {
  createRuntimeCredential,
  listRuntimeCredentials,
  type CreateRuntimeCredentialRequest,
  type RuntimeCredentialMetadata,
} from "../../api/operations";
import {
  deleteProject,
  getProject,
  listProjectRuns,
  MAXIMUM_PROJECT_DESCRIPTION_LENGTH,
  MAXIMUM_PROJECT_NAME_LENGTH,
  normalizeProjectHTTPTarget,
  PROJECT_ID_PATTERN,
  updateProject,
  type Project,
  type ProjectDeletionPhase,
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

type TargetAuthMode = "none" | "existing" | "basic" | "bearer";
type OriginCredential = RuntimeCredentialMetadata & {
  kind: "http-origin-basic@1" | "http-origin-bearer@1";
};

function ProjectHTTPTargetEditor({ project }: { project: Project }) {
  const [editing, setEditing] = useState(false);
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: () =>
      updateProject(api, {
        projectId: project.projectId,
        expectedRevision: project.revision,
        request: { httpTarget: null },
      }),
    onSuccess: async (updated) => {
      queryClient.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      await queryClient.invalidateQueries({
        queryKey: queryKeys.projects.list(project.kind),
      });
    },
  });

  return (
    <div className="project-target-card">
      <div>
        <p className="eyebrow">Allocation-only HTTP configuration</p>
        <h4>Application target</h4>
        {project.httpTarget === undefined ? (
          <p className="muted-copy">
            No target is configured. Workers receive no Project Authorization.
          </p>
        ) : (
          <dl className="metadata-grid project-target-metadata">
            <div>
              <dt>URL</dt>
              <dd>
                <code>{project.httpTarget.url}</code>
              </dd>
            </div>
            <div>
              <dt>Authorization</dt>
              <dd>
                {project.httpTarget.credential === undefined
                  ? "None"
                  : `${project.httpTarget.credential.kind} · ${project.httpTarget.credential.credentialId}`}
              </dd>
            </div>
          </dl>
        )}
      </div>
      {mutation.error === null ? null : (
        <ErrorNotice error={mutation.error} reconcileWrite />
      )}
      <div className="project-form-actions">
        <button
          className="secondary-button"
          type="button"
          onClick={() => setEditing(true)}
        >
          {project.httpTarget === undefined
            ? "Configure target"
            : "Edit target"}
        </button>
        {project.httpTarget === undefined ? null : (
          <button
            className="secondary-button"
            type="button"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate()}
          >
            {mutation.isPending ? "Removing…" : "Remove target"}
          </button>
        )}
      </div>
      {editing ? (
        <ProjectHTTPTargetDialog
          project={project}
          onClose={() => setEditing(false)}
        />
      ) : null}
    </div>
  );
}

function ProjectHTTPTargetDialog({
  project,
  onClose,
}: {
  project: Project;
  onClose: () => void;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const heading = useId();
  const currentCredential = project.httpTarget?.credential;
  const [url, setURL] = useState(project.httpTarget?.url ?? "");
  const [authMode, setAuthMode] = useState<TargetAuthMode>(
    currentCredential === undefined ? "none" : "existing",
  );
  const [credentialID, setCredentialID] = useState(
    currentCredential?.credentialId ?? "",
  );
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [token, setToken] = useState("");
  const [showSecrets, setShowSecrets] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const credentials = useQuery({
    queryKey: queryKeys.operations.runtimeCredentials.list(),
    queryFn: () => listRuntimeCredentials(api),
  });
  const originCredentials = (credentials.data?.items ?? []).filter(
    (credential): credential is OriginCredential =>
      credential.kind === "http-origin-basic@1" ||
      credential.kind === "http-origin-bearer@1",
  );

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape" && !pending) onClose();
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose, pending]);

  function clearSecrets(): void {
    setUsername("");
    setPassword("");
    setToken("");
  }

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setError(null);
    let target;
    try {
      target = normalizeProjectHTTPTarget({ url });
    } catch (reason) {
      setError(reason);
      return;
    }
    setPending(true);
    let createdCredentialID: string | undefined;
    try {
      if (authMode === "existing") {
        const credential =
          originCredentials.find(
            (candidate) => candidate.credentialId === credentialID,
          ) ??
          (currentCredential?.credentialId === credentialID
            ? currentCredential
            : undefined);
        if (credential === undefined) {
          throw new Error("Select an active HTTP origin credential.");
        }
        target = normalizeProjectHTTPTarget({
          url,
          credential: {
            credentialId: credential.credentialId,
            kind: credential.kind,
          },
        });
      } else if (authMode === "basic" || authMode === "bearer") {
        if (authMode === "basic" && (username === "" || password === "")) {
          throw new Error("Username and password are required.");
        }
        if (authMode === "bearer" && token === "") {
          throw new Error("Bearer token is required.");
        }
        const id = `project-target-${crypto.randomUUID()}`;
        const request: CreateRuntimeCredentialRequest =
          authMode === "basic"
            ? {
                credentialId: id,
                kind: "http-origin-basic@1",
                material: { username, password },
              }
            : {
                credentialId: id,
                kind: "http-origin-bearer@1",
                material: { token },
              };
        clearSecrets();
        const credential = await createRuntimeCredential(
          api,
          request,
          `create-project-target-${crypto.randomUUID()}`,
        );
        createdCredentialID = credential.credentialId;
        target = normalizeProjectHTTPTarget({
          url,
          credential: {
            credentialId: credential.credentialId,
            kind:
              authMode === "basic"
                ? "http-origin-basic@1"
                : "http-origin-bearer@1",
          },
        });
      }
      const updated = await updateProject(api, {
        projectId: project.projectId,
        expectedRevision: project.revision,
        request: { httpTarget: target },
      });
      queryClient.setQueryData(
        queryKeys.projects.detail(project.projectId),
        updated,
      );
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: queryKeys.projects.list(project.kind),
        }),
        queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        }),
      ]);
      onClose();
    } catch (reason) {
      if (createdCredentialID !== undefined) {
        setError(
          new Error(
            `Credential ${createdCredentialID} is active, but the Project update failed. Select it under Existing credential and retry.`,
          ),
        );
        setCredentialID(createdCredentialID);
        setAuthMode("existing");
        await queryClient.invalidateQueries({
          queryKey: queryKeys.operations.runtimeCredentials.all,
        });
      } else {
        setError(reason);
      }
    } finally {
      clearSecrets();
      setPending(false);
    }
  }

  return (
    <div className="project-dialog-backdrop" role="presentation">
      <section
        className="project-dialog panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={heading}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Project HTTP target</p>
            <h2 id={heading}>Application access</h2>
          </div>
          <button
            className="project-dialog-close"
            type="button"
            aria-label="Close target dialog"
            disabled={pending}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p className="muted-copy">
          The URL and credential reference are safe metadata. Secret material is
          write-only and reaches only matching HTTP-enabled allocations.
        </p>
        <form
          className="configuration-draft project-target-form"
          autoComplete="off"
          onSubmit={(event) => void submit(event)}
        >
          <label>
            Application URL
            <input
              required
              type="url"
              placeholder="https://app.example.test"
              value={url}
              onChange={(event) => setURL(event.target.value)}
            />
          </label>
          <label>
            Authorization
            <select
              value={authMode}
              onChange={(event) => {
                setAuthMode(event.target.value as TargetAuthMode);
                clearSecrets();
              }}
            >
              <option value="none">No Authorization</option>
              <option value="existing">Existing origin credential</option>
              <option value="basic">New Basic credential</option>
              <option value="bearer">New Bearer credential</option>
            </select>
          </label>
          {authMode === "existing" ? (
            credentials.isPending ? (
              <p className="loading-copy">Loading active credentials…</p>
            ) : credentials.error !== null ? (
              <ErrorNotice error={credentials.error} />
            ) : (
              <label>
                Active HTTP origin credential
                <select
                  required
                  value={credentialID}
                  onChange={(event) => setCredentialID(event.target.value)}
                >
                  <option value="">Select credential</option>
                  {originCredentials.map((credential) => (
                    <option
                      key={credential.credentialId}
                      value={credential.credentialId}
                    >
                      {credential.credentialId} · {credential.kind}
                    </option>
                  ))}
                </select>
              </label>
            )
          ) : authMode === "basic" ? (
            <div className="form-grid">
              <label>
                Username
                <input
                  autoComplete="off"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                />
              </label>
              <label>
                Password · write only
                <input
                  type={showSecrets ? "text" : "password"}
                  autoComplete="new-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </label>
            </div>
          ) : authMode === "bearer" ? (
            <label>
              Bearer token · write only
              <input
                type={showSecrets ? "text" : "password"}
                autoComplete="new-password"
                value={token}
                onChange={(event) => setToken(event.target.value)}
              />
            </label>
          ) : null}
          {authMode === "basic" || authMode === "bearer" ? (
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showSecrets}
                onChange={(event) => setShowSecrets(event.target.checked)}
              />
              Show secret while entering
            </label>
          ) : null}
          {error === null ? null : <ErrorNotice error={error} reconcileWrite />}
          <div className="project-form-actions">
            <button
              type="submit"
              disabled={
                pending || (authMode === "existing" && credentials.isFetching)
              }
            >
              {pending ? "Saving…" : "Save target"}
            </button>
            <button
              className="secondary-button"
              type="button"
              disabled={pending}
              onClick={onClose}
            >
              Cancel
            </button>
          </div>
        </form>
      </section>
    </div>
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

function ProjectAuditsRegion({ projectId }: { projectId: string }) {
  return (
    <ProjectRegion
      eyebrow="Profile-driven verification"
      title="Audits"
      id="project-audits"
      action={
        <Link to={`/projects/${encodeURIComponent(projectId)}/audits`}>
          Open Audits →
        </Link>
      }
    >
      <p className="muted-copy">
        Build a fixed checklist or OpenAPI operation inventory from exact
        Project Artifacts, then follow coverage and ordinary child Runs without
        treating a successful process as a passing assessment.
      </p>
    </ProjectRegion>
  );
}

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
    detail:
      "The remaining ProjectScope history is being purged with reference-safe content cleanup.",
  },
};

function DeleteProjectDialog({
  project,
  pending,
  error,
  onCancel,
  onConfirm,
}: {
  project: Project;
  pending: boolean;
  error: Error | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const warning = useId();
  const [confirmation, setConfirmation] = useState("");
  const resourceLabel = project.kind === "evaluation" ? "Eval" : "Project";
  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent): void {
      if (event.key === "Escape" && !pending) {
        onCancel();
      }
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onCancel, pending]);
  return (
    <div className="project-dialog-backdrop" role="presentation">
      <section
        className="project-dialog project-delete-dialog panel"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={heading}
        aria-describedby={warning}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Permanent workspace deletion</p>
            <h2 id={heading}>Delete {project.name}?</h2>
          </div>
        </div>
        <p className="project-delete-warning" id={warning}>
          This cancels every active Run and permanently deletes all Project
          Runs, execution history, and Project-scoped Artifacts. Shared User
          Artifacts, Skills, and Runtime credentials are retained.
        </p>
        <label>
          Type <strong>{project.name}</strong> to confirm
          <input
            value={confirmation}
            disabled={pending}
            autoComplete="off"
            onChange={(event) => setConfirmation(event.currentTarget.value)}
          />
        </label>
        {error === null ? null : <ErrorNotice error={error} />}
        <div className="run-delete-dialog-actions">
          <button
            className="secondary-button"
            type="button"
            autoFocus
            disabled={pending}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="danger-button"
            type="button"
            disabled={pending || confirmation !== project.name}
            onClick={onConfirm}
          >
            {pending ? "Starting deletion…" : `Delete ${resourceLabel}`}
          </button>
        </div>
      </section>
    </div>
  );
}

function ProjectDeletionProgress({ project }: { project: Project }) {
  const deletion = project.deletion;
  if (project.lifecycle !== "deleting" || deletion === undefined) {
    return null;
  }
  const copy = deletionPhaseCopy[deletion.phase];
  return (
    <div className="panel project-deletion-progress" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <div>
        <p className="eyebrow">Deletion in progress</p>
        <h3>{copy.label}</h3>
        <p>{copy.detail}</p>
        <small>Requested {formatTimestamp(deletion.requestedAt)}</small>
      </div>
      <p className="project-deletion-durability">
        You can leave this page. Cleanup is durable and resumes automatically
        after a Server restart.
      </p>
    </div>
  );
}

function ProjectWorkspaceRoute({
  expectedKind,
}: {
  expectedKind: "project" | "evaluation";
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { projectId = "" } = useParams();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const destination = expectedKind === "evaluation" ? "/evals" : "/projects";
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validProject,
    refetchInterval: (query) =>
      query.state.data?.lifecycle === "deleting" ? 1_000 : false,
    retry: (failureCount, error) =>
      !(error instanceof PublicAPIError && error.status === 404) &&
      failureCount < 2,
  });
  const deletion = useMutation({
    mutationFn: (target: Project) =>
      deleteProject(api, {
        projectId: target.projectId,
        expectedRevision: target.revision,
      }),
    onSuccess: (deleting) => {
      setDeleteOpen(false);
      queryClient.setQueryData(
        queryKeys.projects.detail(deleting.projectId),
        deleting,
      );
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.list(deleting.kind),
      });
    },
  });
  const deletionObserved =
    deletion.data?.lifecycle === "deleting" ||
    project.data?.lifecycle === "deleting";
  const activeProject =
    project.data?.kind === expectedKind && project.data.lifecycle === "active"
      ? project.data
      : undefined;
  const deleteLabel =
    expectedKind === "evaluation" ? "Delete Eval" : "Delete Project";
  useEffect(() => {
    if (
      deletionObserved &&
      project.error instanceof PublicAPIError &&
      project.error.status === 404
    ) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.projects.all });
      void navigate(destination, { replace: true });
    }
  }, [deletionObserved, destination, navigate, project.error, queryClient]);

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
          <Link className="back-link" to={destination}>
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
        <div className="project-header-actions">
          {activeProject !== undefined ? (
            <button
              className="danger-button"
              type="button"
              onClick={() => {
                deletion.reset();
                setDeleteOpen(true);
              }}
            >
              {deleteLabel}
            </button>
          ) : null}
          <button
            className="secondary-button"
            type="button"
            disabled={project.isFetching}
            onClick={() => void project.refetch()}
          >
            {project.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </header>

      {project.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading Project…
        </p>
      ) : deletionObserved &&
        project.error instanceof PublicAPIError &&
        project.error.status === 404 ? (
        <p className="loading-copy" aria-live="polite">
          Project deleted. Returning to{" "}
          {expectedKind === "evaluation" ? "Evals" : "Projects"}…
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
      ) : project.data.lifecycle === "deleting" ? (
        <ProjectDeletionProgress project={project.data} />
      ) : (
        <>
          <nav
            className="project-local-navigation"
            aria-label="Project sections"
          >
            <a href="#project-overview">Overview</a>
            <a href="#project-artifacts">Artifacts</a>
            <a href="#project-workflows">Workflows</a>
            {expectedKind === "project" ? (
              <a href="#project-audits">Audits</a>
            ) : null}
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
            <ProjectHTTPTargetEditor
              key={`target-${project.data.revision}`}
              project={project.data}
            />
          </ProjectRegion>
          <ProjectArtifactRegion
            projectId={project.data.projectId}
            detailRoot={expectedKind === "evaluation" ? "/evals" : "/projects"}
          />
          <ProjectWorkflowRecommendations projectId={project.data.projectId} />
          {expectedKind === "project" ? (
            <ProjectAuditsRegion projectId={project.data.projectId} />
          ) : null}
          <ProjectRunsRegion
            projectId={project.data.projectId}
            evaluation={expectedKind === "evaluation"}
          />
        </>
      )}
      {deleteOpen && activeProject !== undefined ? (
        <DeleteProjectDialog
          project={activeProject}
          pending={deletion.isPending}
          error={deletion.error}
          onCancel={() => {
            if (!deletion.isPending) {
              setDeleteOpen(false);
              deletion.reset();
            }
          }}
          onConfirm={() => deletion.mutate(activeProject)}
        />
      ) : null}
    </section>
  );
}

export function ProjectDetailRoute() {
  return <ProjectWorkspaceRoute expectedKind="project" />;
}

export function EvaluationDetailRoute() {
  return <ProjectWorkspaceRoute expectedKind="evaluation" />;
}
