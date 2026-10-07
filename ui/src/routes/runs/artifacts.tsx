import "./runs.css";
import { ContextLink, ReturnLink } from "../../app/context-navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  downloadRunArtifact,
  getRunArtifactMetadata,
  listRunArtifacts,
  previewRunArtifact,
  RUN_ID_PATTERN,
  type RunStatus,
} from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import { CursorControls } from "../../app/cursor-controls";
import { saveBlob } from "../../app/download";
import { useCursorStack } from "../../app/pagination";
import { ErrorNotice } from "../../app/error-notice";
import { EmptyState, StatusChip } from "../../ui";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import { ArtifactDetailView } from "../artifacts/artifact-detail-view";
import { type RunDisclosureProps, RunSection } from "./components";
import {
  missingOutputCopy,
  organizeRunOutputs,
  outputRole,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
  type OutputEntry,
} from "./output-model";
import { useDocumentTitle } from "../../app/document-title";
import { ArtifactBindingsTable } from "../artifacts/bindings";
import { useNamespaceFilter } from "../artifacts/namespace-filter";
import { artifactDetailPath } from "../artifacts/paths";
import { QueryView } from "../../app/query-view";

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
        {requested && metadata.error === null ? null : (
          <button
            className="ui-btn"
            data-size="sm"
            data-variant={primary ? "primary" : undefined}
            type="button"
            disabled={metadata.isPending && requested}
            onClick={() => {
              if (requested) {
                void metadata.refetch();
              } else {
                setRequested(true);
              }
            }}
          >
            {metadata.isPending && requested
              ? "Loading result…"
              : metadata.error === null
                ? "Preview result"
                : "Retry preview"}
          </button>
        )}
        <button
          className="ui-btn"
          data-size="sm"
          type="button"
          aria-label={
            download.isPending
              ? `Downloading ${entry.slot}…`
              : `Download ${entry.slot}`
          }
          disabled={download.isPending}
          onClick={() => download.mutate()}
        >
          {download.isPending ? "Downloading…" : "Download"}
        </button>
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
          {metadata.isPending ? (
            <p className="runs-loading">Loading output metadata…</p>
          ) : metadata.error !== null ? (
            <ErrorNotice error={metadata.error} />
          ) : (
            <ArtifactPreviewPanel
              archiveScope={{ kind: "run", id: runId }}
              key={metadata.data.artifact.revision}
              metadata={metadata.data}
              unavailableCopy="Inline preview is unavailable for this file. Use Download for the original bytes, or open its details."
              loadPreview={() => previewRunArtifact(api, runId, metadata.data)}
              loadOnMountKey={[
                "run-output-preview",
                runId,
                metadata.data.artifact.namespace,
                metadata.data.artifact.name,
                metadata.data.artifact.revision,
              ]}
            />
          )}
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
        <p className="runs-loading">Loading Workflow output contract…</p>
      ) : null}
      {contract.data === undefined ? null : (
        <p className="runs-hint">
          Primary is the Workflow&apos;s intended display order, not a quality
          mark.
        </p>
      )}
      {contractError === null ? null : (
        <div className="runs-result-contract-error">
          <p className="runs-hint">
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

/** Technical details: every Artifact binding of the Run, by namespace. */
export function RunArtifactLibrary({
  runId,
  ...disclosure
}: { runId: string } & RunDisclosureProps) {
  const api = usePublicAPI();
  const [namespace, setNamespace] = useState<string | undefined>();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.runs.artifacts(runId, namespace, cursor),
    queryFn: () =>
      listRunArtifacts(api, {
        runId,
        ...(namespace === undefined ? {} : { namespace }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  const namespaceFilter = useNamespaceFilter({
    value: namespace,
    onApply: (candidate) => {
      setNamespace(candidate);
      pages.reset();
    },
  });

  const count =
    query.data === undefined
      ? undefined
      : `${query.data.items.length}${query.data.page.hasMore ? "+" : ""}`;
  return (
    <RunSection
      id="run-artifacts"
      title="Run artifacts"
      description="Inputs, intermediate Artifacts and outputs"
      aside={
        count === undefined
          ? undefined
          : `${count} Artifact${count === "1" ? "" : "s"}${namespace === undefined ? "" : ` in ${namespace}`}`
      }
      {...disclosure}
    >
      <div className="runs-namespace-filter">
        {namespaceFilter.form}
        {namespaceFilter.error}
      </div>
      <QueryView
        query={query}
        loading={
          <p className="runs-loading" role="status">
            Loading Run Artifacts…
          </p>
        }
        onRetry={() => void query.refetch()}
        isEmpty={(page) => page.items.length === 0}
        empty={<EmptyState title="No bindings match this view." />}
      >
        {(page) => (
          <ArtifactBindingsTable
            items={page.items}
            returnLabel="Run results"
            showLocked
            detailPath={(item) =>
              artifactDetailPath({ kind: "run", id: runId }, item.artifact)
            }
          />
        )}
      </QueryView>
      <CursorControls
        label="Run Artifact pages"
        {...pages.controls(query.data?.page)}
      />
    </RunSection>
  );
}

export function RunArtifactDetailRoute() {
  const { runId = "", namespace = "", name = "" } = useParams();
  const [searchParams] = useSearchParams();
  const revision = searchParams.get("revision") ?? undefined;
  const valid =
    RUN_ID_PATTERN.test(runId) &&
    ARTIFACT_NAME_PATTERN.test(namespace) &&
    ARTIFACT_NAME_PATTERN.test(name) &&
    (revision === undefined || ARTIFACT_REVISION_PATTERN.test(revision));
  useDocumentTitle(valid ? `${namespace}/${name} · Run` : "Run Artifact");
  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Run Artifact route is invalid")} />
        <Link to="/runs">Return to Runs</Link>
      </section>
    );
  }
  return (
    <ArtifactDetailView
      key={`${runId}/${namespace}/${name}`}
      scope={{ kind: "run", id: runId }}
      namespace={namespace}
      name={name}
      revision={revision}
      className="runs-artifact-page"
      heading={
        <>
          <ReturnLink to={`/runs/${encodeURIComponent(runId)}`} label="Run" />
          <p className="eyebrow">Run inputs</p>
        </>
      }
    />
  );
}
