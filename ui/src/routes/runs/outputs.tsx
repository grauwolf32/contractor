import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  downloadRunArtifact,
  getRunArtifactMetadata,
  isTerminalRunState,
  previewRunArtifact,
  type RunStatus,
} from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import { ContextLink } from "../../app/context-navigation";
import { saveBlob } from "../../app/download";
import { ErrorNotice } from "../../app/error-notice";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { Button, EmptyState, StatusChip } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import {
  missingOutputCopy,
  organizeRunOutputs,
  outputRole,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
  type OutputEntry,
} from "./output-model";
import { publicationStatusLabel } from "./run-state";

function RunOutputPreview({
  runId,
  entry,
  runState,
}: {
  runId: string;
  entry: OutputEntry;
  runState: RunStatus["state"];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [requested, setRequested] = useState(false);
  const titleId = useId();
  const artifact = entry.artifact;
  const metadataKey = queryKeys.runs.artifactMetadata(
    runId,
    artifact?.namespace ?? "missing",
    artifact?.name ?? entry.slot,
    artifact?.revision,
  );
  function readMetadata() {
    if (artifact === undefined) {
      throw new Error("Run output is unavailable");
    }
    return getRunArtifactMetadata(api, { runId, ...artifact });
  }
  const metadata = useQuery({
    queryKey: metadataKey,
    queryFn: readMetadata,
    enabled: requested && artifact !== undefined,
  });
  // Download fallback (US-05): one action reads the exact revision's
  // metadata, then saves its original bytes.
  const download = useMutation({
    mutationFn: async () => {
      const exact = await queryClient.fetchQuery({
        queryKey: metadataKey,
        queryFn: readMetadata,
      });
      return downloadRunArtifact(api, runId, exact);
    },
    onSuccess: (downloaded) => saveBlob(downloaded.blob, downloaded.filename),
  });
  const primary = entry.kind === "primary";
  if (artifact === undefined) {
    return (
      <article
        className={`run-result-card runs-result run-result-${entry.kind}`}
        aria-labelledby={titleId}
      >
        <div className="runs-result-head">
          <div className="runs-result-titles">
            <span className="run-result-role runs-label">
              {outputRole(entry)}
            </span>
            <h3 className="runs-result-title" id={titleId}>
              {entry.slot}
            </h3>
          </div>
          <StatusChip tone="neutral" size="sm" glyph={false}>
            Unavailable
          </StatusChip>
        </div>
        <p className="runs-result-missing">
          {entry.declaration === undefined
            ? "Run output is unavailable."
            : missingOutputCopy(entry.declaration, runState)}
        </p>
      </article>
    );
  }
  const detailPath = artifactDetailPath({ kind: "run", id: runId }, artifact);
  const exactRef = `${artifact.namespace}/${artifact.name}@${artifact.revision}`;

  return (
    <article
      className={`run-result-card runs-result run-result-${entry.kind}`}
      aria-labelledby={titleId}
    >
      <div className="runs-result-head">
        <div className="runs-result-titles">
          <span className="run-result-role runs-label">
            {outputRole(entry)}
          </span>
          <h3 className="runs-result-title" id={titleId}>
            {entry.slot}
          </h3>
          <code className="runs-result-ref">{exactRef}</code>
        </div>
        {primary ? (
          <StatusChip tone="info" size="sm" glyph={false}>
            Primary
          </StatusChip>
        ) : null}
      </div>
      <div className="runs-result-actions">
        {requested ? null : (
          <Button
            size="sm"
            variant={primary ? "primary" : undefined}
            onClick={() => setRequested(true)}
          >
            Preview result
          </Button>
        )}
        <Button
          size="sm"
          aria-label={
            download.isPending
              ? `Downloading ${entry.slot}…`
              : `Download ${entry.slot}`
          }
          pending={download.isPending}
          pendingLabel="Downloading…"
          onClick={() => download.mutate()}
        >
          Download
        </Button>
        <ContextLink
          returnLabel="Run results"
          className="run-output-detail-link runs-result-link"
          to={detailPath}
        >
          Open {exactRef} details →
        </ContextLink>
      </div>
      {download.error === null ? null : (
        <ErrorNotice
          error={download.error}
          context={`Could not download ${entry.slot}`}
        />
      )}
      {requested ? (
        <div className="run-output-preview-body runs-result-preview">
          <QueryView
            query={metadata}
            loading={<p className="ui-caption">Loading output metadata…</p>}
          >
            {(data) => (
              <ArtifactPreviewPanel
                archiveScope={{ kind: "run", id: runId }}
                key={data.artifact.revision}
                metadata={data}
                unavailableCopy="Inline preview is unavailable for this file. Use Download for the original bytes, or open its details."
                loadPreview={() => previewRunArtifact(api, runId, data)}
                loadOnMountKey={[
                  "run-output-preview",
                  runId,
                  data.artifact.namespace,
                  data.artifact.name,
                  data.artifact.revision,
                ]}
              />
            )}
          </QueryView>
        </div>
      ) : null}
    </article>
  );
}

/**
 * The Run's results, primary output first (the Workflow's `primary` flag).
 * Opening the page reads no output bytes: each result previews with one
 * explicit action and can always be downloaded.
 */
export function RunOutputGallery({
  run,
}: {
  run: Pick<RunStatus, "runId" | "workflow" | "state" | "outputs">;
}) {
  const api = usePublicAPI();
  const headingId = useId();
  const identity = parseWorkflowIdentity(run.workflow);
  const contract = useQuery({
    queryKey:
      identity === undefined
        ? ["workflows", "detail", "invalid", run.workflow]
        : queryKeys.workflows.detail(identity.name, identity.version),
    queryFn: async ({ signal }) => {
      if (identity === undefined) {
        throw new Error("Run has an invalid Workflow selector");
      }
      return getWorkflow(api, identity.name, identity.version, signal);
    },
    select: (workflow) => {
      if (identity === undefined) {
        throw new Error("Run has an invalid Workflow selector");
      }
      return requireWorkflowOutputs(workflow, identity);
    },
    enabled: identity !== undefined,
    staleTime: Number.POSITIVE_INFINITY,
  });
  const contractError =
    identity === undefined
      ? new Error("Run has an invalid Workflow selector")
      : contract.error;
  const entries = organizeRunOutputs(run.outputs, contract.data);
  const declaredCount = contract.data
    ? Object.keys(contract.data).length
    : undefined;
  return (
    <section
      className="runs-block-section"
      id="run-outputs"
      aria-labelledby={headingId}
    >
      <div className="runs-section-head">
        <h2 className="runs-h2" id={headingId}>
          Results
        </h2>
        <span className="runs-section-count">
          {Object.keys(run.outputs).length} present
          {declaredCount === undefined ? "" : ` · ${declaredCount} declared`}
        </span>
      </div>
      {contract.isPending && identity !== undefined ? (
        <p className="ui-caption">Loading Workflow output contract…</p>
      ) : null}
      {contract.data === undefined ? null : (
        <p className="ui-caption">
          Primary is the Workflow&apos;s intended display order, not a quality
          mark.
        </p>
      )}
      {contractError === null ? null : (
        <div className="runs-result-contract-error">
          <p className="ui-caption">
            Output roles are unavailable. Present artifacts remain accessible
            without primary classification.
          </p>
          <ErrorNotice error={contractError} />
        </div>
      )}
      {entries.length === 0 ? (
        <EmptyState
          title={
            contract.data === undefined
              ? "No Run outputs are currently available."
              : "This Workflow declares no output slots."
          }
        />
      ) : (
        <div className="runs-result-list">
          {entries.map((entry) => (
            <RunOutputPreview
              key={`${entry.slot}:${entry.artifact?.namespace ?? "missing"}:${entry.artifact?.name ?? "missing"}:${entry.artifact?.revision ?? "missing"}`}
              runId={run.runId}
              runState={run.state}
              entry={entry}
            />
          ))}
        </div>
      )}
    </section>
  );
}

/** Project output publications: which declared outputs the Project kept. */
export function RunOutputPublications({ run }: { run: RunStatus }) {
  const headingId = useId();
  const projectId = run.projectId;
  if (projectId === undefined) {
    return null;
  }
  return (
    <section
      className="runs-block-section runs-publications"
      aria-labelledby={headingId}
    >
      <div className="runs-section-head">
        <div>
          <p className="runs-label">Project outputs</p>
          <h2 className="runs-h2" id={headingId}>
            Reusable output status
          </h2>
        </div>
        <Link
          className="runs-section-link"
          to={`/projects/${encodeURIComponent(projectId)}/artifacts`}
        >
          Open Project
        </Link>
      </div>
      {run.outputPublications.length === 0 ? (
        <p className="ui-caption">
          {isTerminalRunState(run.state)
            ? "No present declared output required a publication receipt."
            : "Publication is recorded only after successful terminal output freezing."}
        </p>
      ) : (
        <ul className="runs-publication-list">
          {run.outputPublications.map((publication) => {
            const status = publicationStatusLabel(publication.status);
            return (
              <li key={`${publication.output}:${publication.source.revision}`}>
                <div className="runs-publication-head">
                  <strong>{publication.output}</strong>
                  <StatusChip tone={status.tone} size="sm">
                    {status.label}
                  </StatusChip>
                  <small>
                    <RecordedTime value={publication.createdAt} />
                  </small>
                </div>
                <span className="runs-publication-ref">
                  source{" "}
                  <code>
                    {publication.source.namespace}/{publication.source.name}@
                    {publication.source.revision}
                  </code>
                </span>
                {publication.target === undefined ? null : (
                  <Link
                    className="runs-publication-ref"
                    to={artifactDetailPath(
                      { kind: "project", id: projectId },
                      publication.target,
                    )}
                  >
                    target {publication.target.namespace}/
                    {publication.target.name}@{publication.target.revision}
                  </Link>
                )}
                {publication.errorMessage === undefined ? null : (
                  <p className="runs-publication-error">
                    <code>{publication.errorCode ?? "publication_failed"}</code>{" "}
                    {publication.errorMessage}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
