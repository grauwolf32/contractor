import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  downloadArtifact,
  getArtifactLineage,
  getArtifactMetadata,
  listArtifactVersions,
  previewArtifact,
  type ArtifactMetadata,
  type DownloadedArtifact,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  ArtifactWriteForm,
  CursorControls,
  ErrorNotice,
  formatBytes,
  formatTimestamp,
} from "./common";
import { ArtifactPreviewPanel } from "./preview";

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

function ArtifactActions({ metadata }: { metadata: ArtifactMetadata }) {
  const api = usePublicAPI();
  const [, setSearchParams] = useSearchParams();
  const download = useMutation({
    mutationFn: () => downloadArtifact(api, metadata),
    onSuccess: triggerDownload,
  });

  return (
    <div className="artifact-actions-grid">
      <ArtifactPreviewPanel
        metadata={metadata}
        unavailableCopy="Inline preview is unavailable for this media type or size. The original bytes are still downloadable."
        loadPreview={() => previewArtifact(api, metadata)}
      />

      <div className="panel artifact-download-panel">
        <p className="eyebrow">Original bytes</p>
        <h3>Download</h3>
        <p className="muted-copy">
          Fetches exactly <code>{metadata.artifact.revision}</code> from Go
          Server; Node does not proxy the payload.
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
          <ArtifactWriteForm
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

function ArtifactHistory({ metadata }: { metadata: ArtifactMetadata }) {
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
    queryKey: queryKeys.artifacts.versions(
      identity.namespace,
      identity.name,
      versionCursor,
    ),
    queryFn: () =>
      listArtifactVersions(api, {
        namespace: identity.namespace,
        name: identity.name,
        ...(versionCursor === undefined ? {} : { cursor: versionCursor }),
      }),
  });
  const lineage = useQuery({
    queryKey: queryKeys.artifacts.lineage(
      identity.namespace,
      identity.name,
      identity.revision,
      lineageCursor,
    ),
    queryFn: () =>
      getArtifactLineage(api, {
        namespace: identity.namespace,
        name: identity.name,
        revision: identity.revision,
        ...(lineageCursor === undefined ? {} : { cursor: lineageCursor }),
      }),
  });

  return (
    <div className="artifact-history-grid">
      <div className="panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Immutable history</p>
            <h3>Versions</h3>
          </div>
        </div>
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
          label="Artifact version pages"
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
            No Run input-fork or output-bind edges reference this revision.
          </div>
        ) : (
          <ol className="lineage-list">
            {lineage.data.items.map((edge, index) => (
              <li
                key={`${edge.kind}-${edge.createdAt}-${edge.source.revision}-${index}`}
              >
                <strong>{edge.kind.replace("_", " ")}</strong>
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
                {edge.stageExecutionId === undefined ? null : (
                  <small>Stage execution {edge.stageExecutionId}</small>
                )}
              </li>
            ))}
          </ol>
        )}
        <CursorControls
          label="Artifact lineage pages"
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

export function ArtifactDetailRoute() {
  const api = usePublicAPI();
  const { namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const validIdentity =
    ARTIFACT_NAME_PATTERN.test(namespace) && ARTIFACT_NAME_PATTERN.test(name);
  const validRevision =
    revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision);
  const query = useQuery({
    queryKey: queryKeys.artifacts.metadata(namespace, name, revision),
    queryFn: () =>
      getArtifactMetadata(api, {
        namespace,
        name,
        ...(revision === undefined ? {} : { revision }),
      }),
    enabled: validIdentity && validRevision,
  });

  if (!validIdentity || !validRevision) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Artifact route is invalid")} />
        <Link to="/artifacts">Return to Artifacts</Link>
      </section>
    );
  }

  return (
    <section className="route-page artifact-page">
      <header className="route-header-row">
        <div>
          <Link className="back-link" to="/artifacts">
            ← All Artifacts
          </Link>
          <p className="eyebrow">UserScope binding</p>
          <h2>
            {namespace}/{name}
          </h2>
          <p className="lede">
            {revision === undefined
              ? "Current authoritative binding"
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
          Loading Artifact metadata…
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
          <ArtifactActions
            key={`actions-${query.data.artifact.revision}`}
            metadata={query.data}
          />
          <ArtifactHistory
            key={`history-${query.data.artifact.revision}`}
            metadata={query.data}
          />
        </>
      )}
    </section>
  );
}
