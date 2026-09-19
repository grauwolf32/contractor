import { Dialog } from "../../app/dialog";
import { ContextLink } from "../../app/context-navigation";
import {
  GitImportDialog,
  GitSourceDetails,
} from "../artifacts/git-import-dialog";
import type { GitImportResult } from "../../api/git-artifacts";
import { useQuery } from "@tanstack/react-query";
import {
  forwardRef,
  type FormEvent,
  useImperativeHandle,
  useState,
  useId,
} from "react";
import { useLocation, Link, useSearchParams } from "react-router";
import {
  ARTIFACT_NAME_PATTERN,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import {
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../artifacts/common";
import {
  ProjectArtifactDialog,
  ProjectArtifactShortcutGrid,
  ProjectRegion,
} from "./common";
import {
  PROJECT_ARTIFACT_SHORTCUTS,
  type ShortcutDefinition,
} from "./shortcuts";

function projectSourceShortcut(): ShortcutDefinition {
  const shortcut = PROJECT_ARTIFACT_SHORTCUTS.find(
    (candidate) => candidate.id === "sources",
  );
  if (shortcut === undefined) {
    throw new Error("Project Sources shortcut is not configured");
  }
  return shortcut;
}

const SOURCE_SHORTCUT = projectSourceShortcut();

export interface ProjectArtifactRegionHandle {
  openSources: () => void;
}

export const ProjectArtifactRegion = forwardRef<
  ProjectArtifactRegionHandle,
  {
    projectId: string;
    detailRoot: "/projects" | "/evals";
    compact?: boolean;
  }
>(function ProjectArtifactRegion(
  { projectId, detailRoot, compact = false },
  ref,
) {
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const addHeading = useId();
  const [addOpen, setAddOpen] = useState(
    () => compact && filters.get("add") === "artifact",
  );
  function closeAdd() {
    setAddOpen(false);
    if (filters.has("add")) {
      const next = new URLSearchParams(filters);
      next.delete("add");
      setFilters(next, { replace: true, state: location.state });
    }
  }
  const namespace = filters.get("artifactsNamespace") || undefined;
  const cursors: Array<string | undefined> = [
    undefined,
    ...filters.getAll("artifactsCursor"),
  ];
  function setCursors(
    update:
      | Array<string | undefined>
      | ((current: Array<string | undefined>) => Array<string | undefined>),
  ) {
    const next = new URLSearchParams(filters);
    next.delete("artifactsCursor");
    for (const cursor of typeof update === "function"
      ? update(cursors)
      : update)
      if (cursor !== undefined) next.append("artifactsCursor", cursor);
    setFilters(next, { preventScrollReset: true, state: location.state });
  }
  function setNamespaceFilter(value: string) {
    const next = new URLSearchParams(filters);
    next.delete("artifactsCursor");
    if (value === "") next.delete("artifactsNamespace");
    else next.set("artifactsNamespace", value);
    setFilters(next, { preventScrollReset: true, state: location.state });
  }
  const [shortcut, setShortcut] = useState<ShortcutDefinition | null>(null);
  const [gitOpen, setGitOpen] = useState(false);
  const [written, setWritten] = useState<
    ArtifactWriteResponse | GitImportResult | null
  >(null);
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

  useImperativeHandle(
    ref,
    () => ({
      openSources(): void {
        setGitOpen(false);
        setShortcut(SOURCE_SHORTCUT);
      },
    }),
    [],
  );

  function applyFilter(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const candidate = String(
      new FormData(event.currentTarget).get("namespaceFilter") ?? "",
    ).trim();
    if (candidate !== "" && !ARTIFACT_NAME_PATTERN.test(candidate)) {
      setFilterError("Namespace filter is not a valid Artifact name.");
      return;
    }
    setFilterError(null);
    setNamespaceFilter(candidate);
  }

  function finishUpload(result: ArtifactWriteResponse): void {
    setWritten(result);
    setShortcut(null);
    setAddOpen(false);
    const next = new URLSearchParams(filters);
    next.delete("add");
    next.delete("artifactsNamespace");
    next.delete("artifactsCursor");
    setFilters(next, { replace: true, state: location.state });
  }

  return (
    <ProjectRegion
      eyebrow="Reusable ProjectScope"
      title="Artifacts"
      id="project-artifacts"
      action={
        <div className="button-row project-region-actions">
          <button
            type="button"
            onClick={() => {
              if (compact) setAddOpen(true);
              else {
                setGitOpen(false);
                setShortcut(SOURCE_SHORTCUT);
              }
            }}
          >
            {compact ? "Add artifact" : "Add sources"}
          </button>
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      }
    >
      {compact ? (
        addOpen ? (
          <Dialog
            className="project-dialog project-add-artifact-dialog panel"
            labelledBy={addHeading}
            onRequestClose={closeAdd}
          >
            <div className="project-dialog-heading">
              <div>
                <p className="eyebrow">Project materials</p>
                <h2 id={addHeading}>Add artifact</h2>
              </div>
              <button
                className="project-dialog-close"
                type="button"
                aria-label="Close artifact choices"
                onClick={closeAdd}
              >
                ×
              </button>
            </div>
            <p className="muted-copy">
              Upload a file or import a Git repository into this project.
            </p>
            <ProjectArtifactShortcutGrid
              onSelect={setShortcut}
              onImportGit={() => setGitOpen(true)}
            />
          </Dialog>
        ) : null
      ) : (
        <>
          <p className="muted-copy">
            Shortcuts suggest useful names and media types. Every field remains
            editable, and Other accepts any supported Artifact.
          </p>
          <ProjectArtifactShortcutGrid
            onSelect={setShortcut}
            onImportGit={() => setGitOpen(true)}
          />
        </>
      )}

      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Project Artifact revision stored.</strong>
          <ContextLink
            returnLabel="Project Artifacts"
            {...(compact ? {} : { returnHash: "#project-artifacts" })}
            to={`${detailRoot}/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(written.artifact.namespace)}/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open {written.artifact.namespace}/{written.artifact.name}@
            {written.artifact.revision}
          </ContextLink>
          {compact ? (
            <Link to={`/projects/${encodeURIComponent(projectId)}/workflows`}>
              Choose a Workflow for this project →
            </Link>
          ) : null}
          {"gitSource" in written ? (
            <GitSourceDetails source={written.gitSource} />
          ) : null}
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
                key={namespace ?? ""}
                defaultValue={namespace ?? ""}
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
            <p>
              {compact
                ? "Add an artifact to prepare the inputs for your next analysis."
                : "Use a shortcut above to add an exact Project input."}
            </p>
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
                      <ContextLink
                        returnLabel="Project Artifacts"
                        {...(compact
                          ? {}
                          : { returnHash: "#project-artifacts" })}
                        to={`${detailRoot}/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}`}
                      >
                        {item.artifact.namespace}/{item.artifact.name}
                      </ContextLink>
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

      {gitOpen ? (
        <GitImportDialog
          projectId={projectId}
          onClose={() => setGitOpen(false)}
          onImported={(result) => {
            finishUpload(result);
            setGitOpen(false);
          }}
        />
      ) : null}
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
});
