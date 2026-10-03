import { Dialog, DialogHeader } from "../../app/dialog";
import {
  GitImportDialog,
  GitSourceDetails,
} from "../artifacts/git-import-dialog";
import type { GitImportResult } from "../../api/git-artifacts";
import { useQuery } from "@tanstack/react-query";
import { useState, useId } from "react";
import { useLocation, Link, useSearchParams } from "react-router";
import { type ArtifactWriteResponse } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { queryKeys } from "../../api/query-keys";
import { CursorControls } from "../../app/cursor-controls";
import { useURLCursorStack } from "../../app/pagination";
import {
  ProjectArtifactDialog,
  ProjectArtifactShortcutGrid,
  ProjectRegion,
} from "./common";
import type { ShortcutDefinition } from "./shortcuts";
import { RefreshButton } from "../../app/refresh-button";
import { QueryView } from "../../app/query-view";
import {
  ArtifactBindingsTable,
  ArtifactStoredNotice,
} from "../artifacts/bindings";
import { useNamespaceFilter } from "../artifacts/namespace-filter";
import { artifactDetailPath } from "../artifacts/paths";

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

  const namespaceFilter = useNamespaceFilter({
    value: namespace,
    onApply: (candidate) => setNamespaceFilter(candidate ?? ""),
  });

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
        <ArtifactStoredNotice
          title="Project Artifact revision stored."
          artifact={written.artifact}
          returnLabel="Project Artifacts"
          to={artifactDetailPath(
            { kind: "project", id: projectId },
            written.artifact,
          )}
        >
          <Link to={`/projects/${encodeURIComponent(projectId)}/workflows`}>
            Choose a Workflow for this project →
          </Link>
          {"gitSource" in written ? (
            <GitSourceDetails source={written.gitSource} />
          ) : null}
        </ArtifactStoredNotice>
      )}

      <div className="project-artifact-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Current bindings</p>
            <h4>Artifact library</h4>
          </div>
          {namespaceFilter.form}
        </div>
        {namespaceFilter.error}
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
            <ArtifactBindingsTable
              items={queryData.items}
              returnLabel="Project Artifacts"
              detailPath={(item) =>
                artifactDetailPath(
                  { kind: "project", id: projectId },
                  {
                    namespace: item.artifact.namespace,
                    name: item.artifact.name,
                  },
                )
              }
            />
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
