import { useMutation, useQuery } from "@tanstack/react-query";
import { type ReactNode, useCallback, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router";

import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { artifactScopeKeys } from "../../api/query-keys";
import { scopedArtifactAPI } from "../../api/scoped-artifacts";
import { CursorControls } from "../../app/cursor-controls";
import { saveBlob } from "../../app/download";
import { ErrorNotice } from "../../app/error-notice";
import { formatBytes } from "../../app/format";
import { useCursorStack } from "../../app/pagination";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import {
  MAXIMUM_SKILL_ARCHIVE_BYTES,
  SKILL_ARCHIVE_MEDIA_TYPE,
} from "./artifact-file";
import { ArtifactWriteForm } from "./common";
import {
  ArtifactHistoryButton,
  ArtifactHistoryDisclosure,
  ArtifactMetadataSummary,
  ArtifactRevisionLede,
} from "./metadata-summary";
import { ArtifactPreviewPanel } from "./preview";

const SCOPE_LABELS = {
  user: "Artifact",
  project: "Project Artifact",
  run: "Run Artifact",
} as const;

/** A version upload the Server rejected, kept visible across the reconcile. */
interface RejectedUpload {
  error: unknown;
  /** The revision named by the upload's If-Match precondition. */
  expectedRevision: string;
}

function ArtifactActions({
  scope,
  metadata,
  onUploadPending,
  onUploadRejected,
}: {
  scope: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
  onUploadPending: (pending: boolean) => void;
  onUploadRejected: (rejected: RejectedUpload) => void;
}) {
  const api = usePublicAPI();
  const location = useLocation();
  const [, setSearchParams] = useSearchParams();
  const artifacts = scopedArtifactAPI(api, scope);
  const download = useMutation({
    mutationFn: () => artifacts.download(metadata),
    onSuccess: (downloaded) => saveBlob(downloaded.blob, downloaded.filename),
  });
  const skillSettings =
    scope.kind === "user" && metadata.artifact.namespace === "skills"
      ? {
          fixedMediaType: SKILL_ARCHIVE_MEDIA_TYPE,
          maximumBytes: MAXIMUM_SKILL_ARCHIVE_BYTES,
        }
      : {};

  return (
    <div
      className={
        scope.kind === "run"
          ? "artifact-actions-grid run-artifact-actions"
          : "artifact-actions-grid"
      }
    >
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
        archiveScope={scope}
        metadata={metadata}
        unavailableCopy="Inline preview is unavailable for this media type or size. The original bytes are still downloadable."
        loadPreview={() => artifacts.preview(metadata)}
      />
      {scope.kind === "run" ? null : metadata.current ? (
        <details className="panel artifact-update-panel">
          <summary>Upload a new version</summary>
          <ArtifactWriteForm
            key={metadata.artifact.revision}
            scope={scope}
            {...skillSettings}
            fixedIdentity={{
              namespace: metadata.artifact.namespace,
              name: metadata.artifact.name,
            }}
            initialMediaType={metadata.mediaType}
            expectedRevision={metadata.artifact.revision}
            onPendingChange={onUploadPending}
            onWriteError={(error) =>
              onUploadRejected({
                error,
                expectedRevision: metadata.artifact.revision,
              })
            }
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

function ArtifactHistory({
  scope,
  metadata,
}: {
  scope: ArtifactArchiveScope;
  metadata: ArtifactMetadata;
}) {
  const location = useLocation();
  const api = usePublicAPI();
  const artifacts = scopedArtifactAPI(api, scope);
  const keys = artifactScopeKeys(scope);
  const label = SCOPE_LABELS[scope.kind];
  const versionPages = useCursorStack();
  const lineagePages = useCursorStack();
  const identity = metadata.artifact;
  const versions = useQuery({
    queryKey: keys.versions(
      identity.namespace,
      identity.name,
      versionPages.cursor,
    ),
    queryFn: () =>
      artifacts.versions({
        namespace: identity.namespace,
        name: identity.name,
        ...(versionPages.cursor === undefined
          ? {}
          : { cursor: versionPages.cursor }),
      }),
  });
  const lineage = useQuery({
    queryKey: keys.lineage(
      identity.namespace,
      identity.name,
      identity.revision,
      lineagePages.cursor,
    ),
    queryFn: () =>
      artifacts.lineage({
        namespace: identity.namespace,
        name: identity.name,
        revision: identity.revision,
        ...(lineagePages.cursor === undefined
          ? {}
          : { cursor: lineagePages.cursor }),
      }),
  });

  return (
    <div className="artifact-history-grid">
      <div className="panel">
        <p className="eyebrow">History</p>
        <h3>Versions</h3>
        <QueryView
          query={versions}
          loading={
            <p className="loading-copy" role="status">
              Loading versions…
            </p>
          }
          onRetry={() => void versions.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={<div className="compact-empty">No versions found.</div>}
        >
          {(page) => (
            <ul className="version-list">
              {page.items.map((item) => (
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
        </QueryView>
        <CursorControls
          label={`${label} version pages`}
          {...versionPages.controls(versions.data?.page)}
        />
      </div>

      <div className="panel">
        <p className="eyebrow">Provenance</p>
        <h3>Lineage</h3>
        <QueryView
          query={lineage}
          loading={
            <p className="loading-copy" role="status">
              Loading lineage…
            </p>
          }
          onRetry={() => void lineage.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={
            <div className="compact-empty">
              No lineage edges reference this revision.
            </div>
          }
        >
          {(page) => (
            <ol className="lineage-list">
              {page.items.map((edge, index) => (
                <li
                  key={`${edge.kind}-${edge.createdAt}-${edge.source.revision}-${index}`}
                >
                  <strong>{edge.kind.replaceAll("_", " ")}</strong>
                  <span>
                    {edge.sourceScope}: {edge.source.namespace}/
                    {edge.source.name}@{edge.source.revision}
                  </span>
                  <span aria-hidden="true">→</span>
                  <span>
                    {edge.targetScope}: {edge.target.namespace}/
                    {edge.target.name}@{edge.target.revision}
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
        </QueryView>
        <CursorControls
          label={`${label} lineage pages`}
          {...lineagePages.controls(lineage.data?.page)}
        />
      </div>
    </div>
  );
}

/**
 * Detail page of one Artifact binding revision in any scope: metadata,
 * download and preview, versions and lineage. User and Project bindings
 * also accept a new version on their current revision.
 */
export function ArtifactDetailView({
  scope,
  namespace,
  name,
  revision,
  className,
  heading,
}: {
  scope: ArtifactArchiveScope;
  namespace: string;
  name: string;
  revision: string | undefined;
  /** Extra class for the page section. */
  className?: string;
  /** Return link and eyebrow above the Artifact title. */
  heading: ReactNode;
}) {
  const api = usePublicAPI();
  const location = useLocation();
  const [, setSearchParams] = useSearchParams();
  const label = SCOPE_LABELS[scope.kind];
  const query = useQuery({
    queryKey: artifactScopeKeys(scope).metadata(namespace, name, revision),
    queryFn: () =>
      scopedArtifactAPI(api, scope).metadata({
        namespace,
        name,
        ...(revision === undefined ? {} : { revision }),
      }),
  });
  // The rejected upload's reconcile remounts the form for a new revision or
  // replaces it on a now historical one, so the notice lives here.
  const [rejectedUpload, setRejectedUpload] = useState<RejectedUpload | null>(
    null,
  );
  const recordUploadPending = useCallback((pending: boolean) => {
    if (pending) setRejectedUpload(null);
  }, []);
  const recordUploadRejected = (rejected: RejectedUpload) => {
    setRejectedUpload(rejected);
    // A precondition conflict means that another writer moved the binding:
    // show its current revision instead of the pinned, now historical one.
    if (
      revision !== undefined &&
      rejected.error instanceof PublicAPIError &&
      rejected.error.code === "conflict"
    ) {
      setSearchParams({}, { replace: true, state: location.state });
    }
  };

  return (
    <section
      className={
        className === undefined
          ? "route-page artifact-page"
          : `route-page artifact-page ${className}`
      }
    >
      <header className="route-header-row">
        <div>
          {heading}
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

      {rejectedUpload === null ? null : (
        <ErrorNotice
          error={rejectedUpload.error}
          context={`The version upload based on ${rejectedUpload.expectedRevision} was rejected`}
          reconcileWrite
        />
      )}
      <QueryView
        query={query}
        loading={
          <p className="loading-copy" role="status">
            {`Loading ${label} metadata…`}
          </p>
        }
        errorContext={`Could not load this ${label}`}
        onRetry={() => void query.refetch()}
      >
        {(metadata) => (
          <>
            <ArtifactMetadataSummary metadata={metadata} />
            <ArtifactActions
              key={`actions-${metadata.artifact.revision}`}
              scope={scope}
              metadata={metadata}
              onUploadPending={recordUploadPending}
              onUploadRejected={recordUploadRejected}
            />
            <ArtifactHistoryDisclosure>
              <ArtifactHistory
                key={`history-${metadata.artifact.revision}`}
                scope={scope}
                metadata={metadata}
              />
            </ArtifactHistoryDisclosure>
          </>
        )}
      </QueryView>
    </section>
  );
}
