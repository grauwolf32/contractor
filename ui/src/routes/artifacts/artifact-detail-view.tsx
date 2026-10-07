import { useMutation, useQuery } from "@tanstack/react-query";
import { type ReactNode, useCallback, useId, useRef, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router";

import type { ArtifactArchiveScope } from "../../api/artifact-archive";
import type { ArtifactMetadata } from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { artifactScopeKeys } from "../../api/query-keys";
import {
  scopedArtifactAPI,
  type WritableArtifactScope,
} from "../../api/scoped-artifacts";
import { saveBlob } from "../../app/download";
import { ErrorNotice } from "../../app/error-notice";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { StatusGlyph } from "../../ui";
import {
  MAXIMUM_SKILL_ARCHIVE_BYTES,
  SKILL_ARCHIVE_MEDIA_TYPE,
} from "./artifact-file";
import { ArtifactWriteForm } from "./common";
import { MaterialIcon, MaterialKindIcon } from "./icons";
import { materialKindLabel, materialKindOf } from "./kinds";
import { MaterialName } from "./material-row";
import {
  ArtifactFacts,
  ArtifactRevisionLede,
  ArtifactTechnicalDetails,
  ArtifactVersionsButton,
} from "./metadata-summary";
import { ArtifactPreviewPanel } from "./preview";
import "./materials.css";

/** What one binding is called in each scope's copy. */
const SCOPE_NOUNS = {
  user: "file",
  project: "material",
  run: "file",
} as const;

/** A version upload the Server rejected, kept visible across the reconcile. */
interface RejectedUpload {
  error: unknown;
  /** The revision named by the upload's If-Match precondition. */
  expectedRevision: string;
}

/**
 * "Upload a new version" of the current revision. The form is keyed by the
 * revision it replaces (If-Match), so a reconcile to a newer current
 * revision starts a fresh form while the panel stays open.
 */
function ArtifactUpdatePanel({
  scope,
  metadata,
  onUploadPending,
  onUploadRejected,
}: {
  scope: WritableArtifactScope;
  metadata: ArtifactMetadata;
  onUploadPending: (pending: boolean) => void;
  onUploadRejected: (rejected: RejectedUpload) => void;
}) {
  const location = useLocation();
  const [, setSearchParams] = useSearchParams();
  const summaryId = useId();
  const panel = useRef<HTMLDetailsElement>(null);
  const skillSettings =
    scope.kind === "user" && metadata.artifact.namespace === "skills"
      ? {
          fixedMediaType: SKILL_ARCHIVE_MEDIA_TYPE,
          maximumBytes: MAXIMUM_SKILL_ARCHIVE_BYTES,
        }
      : {};
  return (
    <details ref={panel} className="materials-update">
      <summary id={summaryId}>
        <MaterialIcon name="upload" />
        Upload a new version
      </summary>
      <div className="materials-update-body">
        <p className="materials-quiet">
          The file becomes the current version. Earlier versions stay in the
          history, and checks and Runs keep the version they read.
        </p>
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
          headingId={summaryId}
          onPendingChange={onUploadPending}
          onWriteError={(error) =>
            onUploadRejected({
              error,
              expectedRevision: metadata.artifact.revision,
            })
          }
          onWritten={(result) => {
            if (panel.current !== null) panel.current.open = false;
            setSearchParams(
              { revision: result.artifact.revision },
              { state: location.state },
            );
          }}
        />
      </div>
    </details>
  );
}

/** An earlier revision: read-only, with the way back to the current one. */
function HistoricalNote() {
  const location = useLocation();
  return (
    <p className="materials-callout materials-historical">
      <StatusGlyph tone="idle" size={14} />
      Historical revisions are read-only.{" "}
      <Link to={{ search: "" }} state={location.state}>
        Open the current version
      </Link>
    </p>
  );
}

/**
 * Detail page of one binding revision in any scope: header with the kind,
 * name and whether the revision is current, download and versions; facts
 * (format, size, creation, lock, Git source); a new version for the current
 * revision of a User or Project binding; the bounded preview or archive
 * browser; and revision, media type, versions and lineage behind Technical
 * details.
 */
export function ArtifactDetailView({
  scope,
  namespace,
  name,
  revision,
  className,
  heading,
  variant,
}: {
  scope: ArtifactArchiveScope;
  namespace: string;
  name: string;
  revision: string | undefined;
  /** Extra class for the page section. */
  className?: string | undefined;
  /** Return link (and an optional eyebrow) above the title. */
  heading: ReactNode;
  /** Shows the kind ("Source code", "Skill package") above the title. */
  variant?: "material" | "file" | undefined;
}) {
  const api = usePublicAPI();
  const location = useLocation();
  const [, setSearchParams] = useSearchParams();
  const titleId = useId();
  const noun = SCOPE_NOUNS[scope.kind];
  const artifacts = scopedArtifactAPI(api, scope);
  const query = useQuery({
    queryKey: artifactScopeKeys(scope).metadata(namespace, name, revision),
    queryFn: () =>
      scopedArtifactAPI(api, scope).metadata({
        namespace,
        name,
        ...(revision === undefined ? {} : { revision }),
      }),
  });
  const metadata = query.data;
  const download = useMutation({
    mutationFn: (shown: ArtifactMetadata) => artifacts.download(shown),
    onSuccess: (downloaded) => saveBlob(downloaded.blob, downloaded.filename),
  });
  // Download state belongs to the revision it was started for.
  const downloadFor = download.variables?.artifact.revision;
  const downloadShown =
    metadata !== undefined && downloadFor === metadata.artifact.revision;
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
      className={[
        "route-page artifact-page materials-page materials-detail",
        className,
      ]
        .filter(Boolean)
        .join(" ")}
      aria-labelledby={titleId}
    >
      <header className="materials-detail-header">
        <div className="materials-detail-context">{heading}</div>
        {variant === undefined || metadata === undefined ? null : (
          <p className="materials-kicker">
            <MaterialKindIcon kind={materialKindOf(metadata)} size={14} />
            {materialKindLabel(metadata)}
          </p>
        )}
        <div className="materials-detail-titlebar">
          <div className="materials-detail-titles">
            <h1 id={titleId} className="materials-detail-title">
              <MaterialName namespace={namespace} name={name} />
            </h1>
            <ArtifactRevisionLede metadata={metadata} />
          </div>
          <div className="materials-actions">
            {metadata === undefined ? null : (
              <>
                <button
                  className="ui-btn"
                  data-size="sm"
                  type="button"
                  disabled={downloadShown && download.isPending}
                  onClick={() => download.mutate(metadata)}
                >
                  <MaterialIcon name="download" />
                  {downloadShown && download.isPending
                    ? "Downloading…"
                    : "Download this revision"}
                </button>
                <ArtifactVersionsButton />
              </>
            )}
            <RefreshButton
              className="ui-btn materials-refresh"
              isFetching={query.isFetching}
              onRefresh={() => void query.refetch()}
            />
          </div>
        </div>
      </header>

      <div className="materials-detail-body">
        {downloadShown && download.error !== null ? (
          <ErrorNotice
            error={download.error}
            context="The download did not finish"
          />
        ) : null}
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
            <p className="materials-loading" role="status">
              {`Loading this ${noun}…`}
            </p>
          }
          errorContext={`Could not load this ${noun}`}
          onRetry={() => void query.refetch()}
        >
          {(shown) => (
            <>
              <ArtifactFacts metadata={shown} />
              {scope.kind === "run" ? null : shown.current ? (
                <ArtifactUpdatePanel
                  scope={scope}
                  metadata={shown}
                  onUploadPending={recordUploadPending}
                  onUploadRejected={recordUploadRejected}
                />
              ) : (
                <HistoricalNote />
              )}
              <ArtifactPreviewPanel
                key={`preview-${shown.artifact.revision}`}
                archiveScope={scope}
                metadata={shown}
                titleAs="h2"
                unavailableCopy="Inline preview is unavailable for this format or size. Download this revision to open it."
                loadPreview={() => artifacts.preview(shown)}
              />
              <ArtifactTechnicalDetails
                scope={scope}
                metadata={shown}
                titleAs="h2"
              />
            </>
          )}
        </QueryView>
      </div>
    </section>
  );
}
