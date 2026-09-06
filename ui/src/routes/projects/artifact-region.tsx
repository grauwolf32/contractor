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
} from "react";
import { Link } from "react-router";
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
  }
>(function ProjectArtifactRegion({ projectId, detailRoot }, ref) {
  const api = usePublicAPI();
  const [namespaceDraft, setNamespaceDraft] = useState("");
  const [namespace, setNamespace] = useState<string | undefined>();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
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
      <ProjectArtifactShortcutGrid
        onSelect={setShortcut}
        onImportGit={() => setGitOpen(true)}
      />

      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Project Artifact revision stored.</strong>
          <Link
            to={`${detailRoot}/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(written.artifact.namespace)}/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open {written.artifact.namespace}/{written.artifact.name}@
            {written.artifact.revision}
          </Link>
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
