import { Dialog, DialogHeader } from "../../app/dialog";
import { ContextLink } from "../../app/context-navigation";
import {
  GitImportDialog,
  GitSourceDetails,
} from "../artifacts/git-import-dialog";
import type { GitImportResult } from "../../api/git-artifacts";
import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useState, useId } from "react";
import { useLocation, Link, useSearchParams } from "react-router";
import {
  ARTIFACT_NAME_PATTERN,
  type ArtifactWriteResponse,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { useURLCursorStack } from "../../app/pagination";
import { formatBytes, formatTimestamp } from "../../app/format";
import {
  ProjectArtifactDialog,
  ProjectArtifactShortcutGrid,
  ProjectRegion,
} from "./common";
import type { ShortcutDefinition } from "./shortcuts";
import { RefreshButton } from "../../app/refresh-button";
import { QueryView } from "../../app/query-view";

export function ProjectArtifactRegion({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const addHeading = useId();
  const [addOpen, setAddOpen] = useState(
    () => filters.get("add") === "artifact",
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
  const pages = useURLCursorStack({
    param: "artifactsCursor",
    navigateOptions: { preventScrollReset: true, state: location.state },
  });
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
  const cursor = pages.cursor;
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
      eyebrow="Project artifacts"
      title="Artifacts"
      id="project-artifacts"
      compact
      action={
        <div className="button-row project-region-actions">
          <button type="button" onClick={() => setAddOpen(true)}>
            Add artifact
          </button>
          <RefreshButton
            isFetching={query.isFetching}
            onRefresh={() => void query.refetch()}
            label="Refresh"
          />
        </div>
      }
    >
      {addOpen ? (
        <Dialog
          className="project-dialog project-add-artifact-dialog panel"
          labelledBy={addHeading}
          onRequestClose={closeAdd}
        >
          <DialogHeader
            id={addHeading}
            eyebrow="Project materials"
            title="Add artifact"
            close={{ label: "Close artifact choices", onClose: closeAdd }}
          />
          <p className="muted-copy">
            Upload a file or import a Git repository into this project.
          </p>
          <ProjectArtifactShortcutGrid
            onSelect={setShortcut}
            onImportGit={() => setGitOpen(true)}
          />
        </Dialog>
      ) : null}

      {written === null ? null : (
        <div className="notice notice-success" role="status">
          <strong>Project Artifact revision stored.</strong>
          <ContextLink
            returnLabel="Project Artifacts"
            to={`/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(written.artifact.namespace)}/${encodeURIComponent(written.artifact.name)}?revision=${encodeURIComponent(written.artifact.revision)}`}
          >
            Open {written.artifact.namespace}/{written.artifact.name}@
            {written.artifact.revision}
          </ContextLink>
          <Link to={`/projects/${encodeURIComponent(projectId)}/workflows`}>
            Choose a Workflow for this project →
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
        <QueryView
          query={query}
          loading={
            <p className="loading-copy" role="status">
              Loading Project Artifacts…
            </p>
          }
          onRetry={() => void query.refetch()}
          isEmpty={(queryData) => queryData.items.length === 0}
          empty={
            <div className="compact-empty">
              <strong>No Artifact bindings in this view.</strong>
              <p>
                Add an artifact to prepare the inputs for your next analysis.
              </p>
            </div>
          }
        >
          {(queryData) => (
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
                  {queryData.items.map((item) => (
                    <tr
                      key={`${item.artifact.namespace}/${item.artifact.name}`}
                    >
                      <td data-label="Binding">
                        <ContextLink
                          returnLabel="Project Artifacts"
                          to={`/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}`}
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
        </QueryView>
        <CursorControls
          label="Project Artifact pages"
          {...pages.controls(query.data?.page)}
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
}
