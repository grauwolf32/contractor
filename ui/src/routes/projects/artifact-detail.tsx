import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  type ArtifactMetadata,
  type DownloadedArtifact,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import {
  downloadProjectArtifact,
  getProjectArtifactLineage,
  getProjectArtifactMetadata,
  listProjectArtifactVersions,
  previewProjectArtifact,
} from "../../api/project-artifacts";
import { PROJECT_ID_PATTERN } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import {
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "../artifacts/common";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import { ProjectArtifactWriteForm } from "./common";

function triggerDownload(downloaded: DownloadedArtifact): void {
  const objectURL = URL.createObjectURL(downloaded.blob);
  const anchor = document.createElement("a");
  anchor.href = objectURL;
  anchor.download = downloaded.filename;
  anchor.hidden = true;
  document.body.append(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(objectURL);
  }
}

function ProjectArtifactActions({
  projectId,
  metadata,
}: {
  projectId: string;
  metadata: ArtifactMetadata;
}) {
  const api = usePublicAPI();
  const [, setSearchParams] = useSearchParams();
  const download = useMutation({
    mutationFn: () => downloadProjectArtifact(api, projectId, metadata),
    onSuccess: triggerDownload,
  });

  return (
    <div className="artifact-actions-grid">
      <ArtifactPreviewPanel
        metadata={metadata}
        unavailableCopy="Inline preview is unavailable for this media type or size. The original bytes are still downloadable."
        loadPreview={() => previewProjectArtifact(api, projectId, metadata)}
      />
      <div className="panel artifact-download-panel">
        <p className="eyebrow">Original bytes</p>
        <h3>Download</h3>
        <p className="muted-copy">
          Fetches exact ProjectScope revision{" "}
          <code>{metadata.artifact.revision}</code> directly from Go Server.
        </p>
        {download.error === null ? null : (
          <ErrorNotice error={download.error} />
        )}
        <button
          type="button"
          disabled={download.isPending}
          onClick={() => download.mutate()}
        >
          {download.isPending ? "Downloading…" : "Download exact revision"}
        </button>
      </div>
      {metadata.current ? (
        <div className="panel artifact-update-panel">
          <ProjectArtifactWriteForm
            projectId={projectId}
            fixedIdentity={{
              namespace: metadata.artifact.namespace,
              name: metadata.artifact.name,
            }}
            expectedRevision={metadata.artifact.revision}
            onWritten={(result) =>
              setSearchParams({ revision: result.artifact.revision })
            }
          />
        </div>
      ) : (
        <div className="panel compact-empty">
          <strong>Historical revision is immutable.</strong>
          <p>
            Select the current revision before starting an exact CAS update.
          </p>
        </div>
      )}
    </div>
  );
}

function ProjectArtifactHistory({
  projectId,
  metadata,
}: {
  projectId: string;
  metadata: ArtifactMetadata;
}) {
  const api = usePublicAPI();
  const [versionCursors, setVersionCursors] = useState<
    Array<string | undefined>
  >([undefined]);
  const [lineageCursors, setLineageCursors] = useState<
    Array<string | undefined>
  >([undefined]);
  const versionCursor = versionCursors.at(-1);
  const lineageCursor = lineageCursors.at(-1);
  const identity = metadata.artifact;
  const versions = useQuery({
    queryKey: queryKeys.projects.artifacts.versions(
      projectId,
      identity.namespace,
      identity.name,
      versionCursor,
    ),
    queryFn: () =>
      listProjectArtifactVersions(api, {
        projectId,
        namespace: identity.namespace,
        name: identity.name,
        ...(versionCursor === undefined ? {} : { cursor: versionCursor }),
      }),
  });
  const lineage = useQuery({
    queryKey: queryKeys.projects.artifacts.lineage(
      projectId,
      identity.namespace,
      identity.name,
      identity.revision,
      lineageCursor,
    ),
    queryFn: () =>
      getProjectArtifactLineage(api, {
        projectId,
        namespace: identity.namespace,
        name: identity.name,
        revision: identity.revision,
        ...(lineageCursor === undefined ? {} : { cursor: lineageCursor }),
      }),
  });

  return (
    <div className="artifact-history-grid">
      <div className="panel">
        <p className="eyebrow">Immutable history</p>
        <h3>Versions</h3>
        {versions.isPending ? (
          <p className="loading-copy">Loading versions…</p>
        ) : versions.error !== null ? (
          <ErrorNotice error={versions.error} />
        ) : versions.data.items.length === 0 ? (
          <div className="compact-empty">No versions found.</div>
        ) : (
          <ul className="version-list">
            {versions.data.items.map((item) => (
              <li
                key={item.artifact.revision}
                className={
                  item.artifact.revision === identity.revision
                    ? "selected"
                    : undefined
                }
              >
                <Link
                  to={`?revision=${encodeURIComponent(item.artifact.revision)}`}
                >
                  <code>{item.artifact.revision}</code>
                  <span>{formatBytes(item.size)}</span>
                  <span>{formatTimestamp(item.createdAt)}</span>
                  {item.current ? <strong>current</strong> : null}
                </Link>
              </li>
            ))}
          </ul>
        )}
        <CursorControls
          label="Project Artifact version pages"
          canGoBack={versionCursors.length > 1}
          {...(versions.data?.page.hasMore === true &&
          versions.data.page.nextCursor !== undefined
            ? { nextCursor: versions.data.page.nextCursor }
            : {})}
          onBack={() =>
            setVersionCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setVersionCursors((current) => [...current, next])}
        />
      </div>

      <div className="panel">
        <p className="eyebrow">Provenance</p>
        <h3>Lineage</h3>
        {lineage.isPending ? (
          <p className="loading-copy">Loading lineage…</p>
        ) : lineage.error !== null ? (
          <ErrorNotice error={lineage.error} />
        ) : lineage.data.items.length === 0 ? (
          <div className="compact-empty">
            No Run input-fork or output-publication edges reference this
            revision.
          </div>
        ) : (
          <ol className="lineage-list">
            {lineage.data.items.map((edge, index) => (
              <li
                key={`${edge.kind}-${edge.createdAt}-${edge.source.revision}-${index}`}
              >
                <strong>{edge.kind.replaceAll("_", " ")}</strong>
                <span>
                  {edge.sourceScope}: {edge.source.namespace}/{edge.source.name}
                  @{edge.source.revision}
                </span>
                <span aria-hidden="true">→</span>
                <span>
                  {edge.targetScope}: {edge.target.namespace}/{edge.target.name}
                  @{edge.target.revision}
                </span>
                {edge.runId === undefined ? null : (
                  <small>Run {edge.runId}</small>
                )}
              </li>
            ))}
          </ol>
        )}
        <CursorControls
          label="Project Artifact lineage pages"
          canGoBack={lineageCursors.length > 1}
          {...(lineage.data?.page.hasMore === true &&
          lineage.data.page.nextCursor !== undefined
            ? { nextCursor: lineage.data.page.nextCursor }
            : {})}
          onBack={() =>
            setLineageCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setLineageCursors((current) => [...current, next])}
        />
      </div>
    </div>
  );
}

function ProjectArtifactDetailRouteView({
  detailRoot,
}: {
  detailRoot: "/projects" | "/evals";
}) {
  const api = usePublicAPI();
  const { projectId = "", namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const validRoute =
    PROJECT_ID_PATTERN.test(projectId) &&
    ARTIFACT_NAME_PATTERN.test(namespace) &&
    ARTIFACT_NAME_PATTERN.test(name) &&
    (revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision));
  const query = useQuery({
    queryKey: queryKeys.projects.artifacts.metadata(
      projectId,
      namespace,
      name,
      revision,
    ),
    queryFn: () =>
      getProjectArtifactMetadata(api, {
        projectId,
        namespace,
        name,
        ...(revision === undefined ? {} : { revision }),
      }),
    enabled: validRoute,
  });

  if (!validRoute) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Project Artifact route is invalid")} />
        <Link to={detailRoot}>
          Return to {detailRoot === "/evals" ? "Evals" : "Projects"}
        </Link>
      </section>
    );
  }

  return (
    <section className="route-page artifact-page project-artifact-page">
      <header className="route-header-row">
        <div>
          <Link
            className="back-link"
            to={`${detailRoot}/${encodeURIComponent(projectId)}#project-artifacts`}
          >
            ← {detailRoot === "/evals" ? "Eval" : "Project"} Artifacts
          </Link>
          <p className="eyebrow">ProjectScope binding</p>
          <h2>
            {namespace}/{name}
          </h2>
          <p className="lede">
            {revision === undefined
              ? "Current authoritative Project binding"
              : "Selected immutable historical revision"}
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading Project Artifact metadata…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <>
          <dl className="metadata-grid panel">
            <div>
              <dt>Exact revision</dt>
              <dd>
                <code>{query.data.artifact.revision}</code>
              </dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd>{query.data.current ? "current" : "historical"}</dd>
            </div>
            <div>
              <dt>Media type</dt>
              <dd>{query.data.mediaType}</dd>
            </div>
            <div>
              <dt>Size</dt>
              <dd>{formatBytes(query.data.size)}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{formatTimestamp(query.data.createdAt)}</dd>
            </div>
            <div>
              <dt>Frozen</dt>
              <dd>{query.data.frozen ? "yes" : "no"}</dd>
            </div>
          </dl>
          <ProjectArtifactActions
            key={`actions-${query.data.artifact.revision}`}
            projectId={projectId}
            metadata={query.data}
          />
          <ProjectArtifactHistory
            key={`history-${query.data.artifact.revision}`}
            projectId={projectId}
            metadata={query.data}
          />
        </>
      )}
    </section>
  );
}

export function ProjectArtifactDetailRoute() {
  return <ProjectArtifactDetailRouteView detailRoot="/projects" />;
}

export function EvaluationArtifactDetailRoute() {
  return <ProjectArtifactDetailRouteView detailRoot="/evals" />;
}
