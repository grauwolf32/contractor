import {
  ArtifactMetadataSummary,
  ArtifactHistoryDisclosure,
  ArtifactHistoryButton,
  ArtifactRevisionLede,
} from "./metadata-summary";
import { useDocumentTitle } from "../../app/document-title";
import { ReturnLink } from "../../app/context-navigation";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useLocation, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
  downloadArtifact,
  getArtifactLineage,
  getArtifactMetadata,
  listArtifactVersions,
  previewArtifact,
  type ArtifactMetadata,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { ArtifactWriteForm } from "./common";
import { CursorControls } from "../../app/cursor-controls";
import { ErrorNotice } from "../../app/error-notice";
import { formatBytes } from "../../app/format";
import { ArtifactPreviewPanel } from "./preview";
import {
  MAXIMUM_SKILL_ARCHIVE_BYTES,
  SKILL_ARCHIVE_MEDIA_TYPE,
} from "./artifact-file";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { RecordedTime } from "../../app/recorded-time";
import { saveBlob } from "../../app/download";

function ArtifactActions({ metadata }: { metadata: ArtifactMetadata }) {
  const api = usePublicAPI();
  const location = useLocation();
  const [, setSearchParams] = useSearchParams();
  const skillSettings =
    metadata.artifact.namespace === "skills"
      ? {
          fixedMediaType: SKILL_ARCHIVE_MEDIA_TYPE,
          maximumBytes: MAXIMUM_SKILL_ARCHIVE_BYTES,
        }
      : {};
  const download = useMutation({
    mutationFn: () => downloadArtifact(api, metadata),
    onSuccess: (downloaded) => saveBlob(downloaded.blob, downloaded.filename),
  });

  return (
    <div className="artifact-actions-grid">
      <div className="artifact-file-toolbar">
        {download.error === null ? null : (
          <ErrorNotice error={download.error} />
        )}
        <button
          className="secondary-button"
          type="button"
          disabled={download.isPending}
          onClick={() => download.mutate()}
        >
          {download.isPending ? "Downloading…" : "Download this revision"}
        </button>
        <ArtifactHistoryButton />
      </div>
      <ArtifactPreviewPanel
        archiveScope={{ kind: "user" }}
        metadata={metadata}
        unavailableCopy="Inline preview is unavailable for this media type or size. The original bytes are still downloadable."
        loadPreview={() => previewArtifact(api, metadata)}
      />

      {metadata.current ? (
        <details className="panel artifact-update-panel">
          <summary>Upload a new version</summary>
          <ArtifactWriteForm
            key={metadata.artifact.revision}
            {...skillSettings}
            fixedIdentity={{
              namespace: metadata.artifact.namespace,
              name: metadata.artifact.name,
            }}
            initialMediaType={metadata.mediaType}
            expectedRevision={metadata.artifact.revision}
            onWritten={(result) =>
              setSearchParams(
                { revision: result.artifact.revision },
                { state: location.state },
              )
            }
          />
        </details>
      ) : (
        <div className="panel compact-empty">
          <strong>Historical revisions are read-only.</strong>
          <p>Select the current revision to upload a new version.</p>
        </div>
      )}
    </div>
  );
}

function ArtifactHistory({ metadata }: { metadata: ArtifactMetadata }) {
  const location = useLocation();
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
            <p className="eyebrow">History</p>
            <h3>Versions</h3>
          </div>
        </div>
        {versions.isPending ? (
          <p className="loading-copy" role="status">
            Loading versions…
          </p>
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
                  state={location.state}
                >
                  <code>{item.artifact.revision}</code>
                  <span>{formatBytes(item.size)}</span>
                  <RecordedTime value={item.createdAt} />
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
          <p className="loading-copy" role="status">
            Loading lineage…
          </p>
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
  useDocumentTitle(name ? `${namespace}/${name}` : "Artifact");
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
          <ReturnLink
            to={namespace === "skills" ? "/catalog/skills" : "/artifacts"}
            label={namespace === "skills" ? "Skills" : "All Artifacts"}
          />

          <h2>
            {namespace}/{name}
          </h2>
          <ArtifactRevisionLede
            metadata={query.isSuccess ? query.data : undefined}
          />
        </div>
        <RefreshButton
          isFetching={query.isFetching}
          onRefresh={() => void query.refetch()}
          label="Refresh"
        />
      </header>

      <QueryView
        query={query}
        loading={
          <p className="loading-copy" role="status">
            Loading Artifact metadata…
          </p>
        }
        errorContext="Could not load this Artifact"
        onRetry={() => void query.refetch()}
      >
        {(metadata) => (
          <>
            <ArtifactMetadataSummary metadata={metadata} />
            <ArtifactActions
              key={`actions-${metadata.artifact.revision}`}
              metadata={metadata}
            />
            <ArtifactHistoryDisclosure>
              <ArtifactHistory
                key={`history-${metadata.artifact.revision}`}
                metadata={metadata}
              />
            </ArtifactHistoryDisclosure>
          </>
        )}
      </QueryView>
    </section>
  );
}
