import { ContextLink, ReturnLink } from "../../app/context-navigation";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import {
  ARTIFACT_NAME_PATTERN,
  ARTIFACT_REVISION_PATTERN,
} from "../../api/artifacts";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  getRunArtifactMetadata,
  listRunArtifacts,
  previewRunArtifact,
  RUN_ID_PATTERN,
  type RunStatus,
} from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import { CursorControls } from "../../app/cursor-controls";
import { useCursorStack } from "../../app/pagination";
import { ErrorNotice } from "../../app/error-notice";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import { ArtifactDetailView } from "../artifacts/artifact-detail-view";
import { type RunDisclosureProps, RunDisclosureSummary } from "./components";
import {
  missingOutputCopy,
  organizeRunOutputs,
  outputRole,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
  type OutputEntry,
} from "./output-model";
import "./outputs.css";
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
  const [requested, setRequested] = useState(false);
  const artifact = entry.artifact;
  const metadata = useQuery({
    queryKey: queryKeys.runs.artifactMetadata(
      runId,
      artifact?.namespace ?? "missing",
      artifact?.name ?? entry.slot,
      artifact?.revision,
    ),
    queryFn: () => {
      if (artifact === undefined) {
        throw new Error("Run output is unavailable");
      }
      return getRunArtifactMetadata(api, { runId, ...artifact });
    },
    enabled: requested && artifact !== undefined,
  });
  if (artifact === undefined) {
    return (
      <article className={`run-result-card run-result-${entry.kind}`}>
        <div className="run-result-heading">
          <div>
            <span className="run-result-role">{outputRole(entry)}</span>
            <h4>{entry.slot}</h4>
          </div>
          <span className="run-result-state">Unavailable</span>
        </div>
        <p className="run-result-missing">
          {entry.declaration === undefined
            ? "Run output is unavailable."
            : missingOutputCopy(entry.declaration, runState)}
        </p>
      </article>
    );
  }
  const detailPath = artifactDetailPath({ kind: "run", id: runId }, artifact);

  return (
    <article className={`run-result-card run-result-${entry.kind}`}>
      <div className="run-result-heading">
        <div>
          <span className="run-result-role">{outputRole(entry)}</span>
          <h4>{entry.slot}</h4>
          <code>
            {artifact.namespace}/{artifact.name}@{artifact.revision}
          </code>
        </div>
        {entry.kind === "primary" ? (
          <span className="run-result-state">Primary</span>
        ) : null}
      </div>
      <div className="run-result-actions">
        {requested && metadata.error === null ? null : (
          <button
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
        <ContextLink
          returnLabel="Run results"
          className="run-output-detail-link"
          to={detailPath}
        >
          Open {artifact.namespace}/{artifact.name}@{artifact.revision} details
          →
        </ContextLink>
      </div>
      {requested ? (
        <div className="run-output-preview-body">
          {!requested || metadata.isPending ? (
            <p className="loading-copy">Loading output metadata…</p>
          ) : metadata.error !== null ? (
            <ErrorNotice error={metadata.error} />
          ) : (
            <ArtifactPreviewPanel
              archiveScope={{ kind: "run", id: runId }}
              key={metadata.data.artifact.revision}
              metadata={metadata.data}
              unavailableCopy="Inline preview is unavailable; the original file remains available from artifact details."
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

export function RunOutputGallery({
  run,
}: {
  run: Pick<RunStatus, "runId" | "workflow" | "state" | "outputs">;
}) {
  const api = usePublicAPI();
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
    <section className="run-output-gallery" id="run-outputs">
      <div className="section-heading">
        <div>
          <h3>Results</h3>
        </div>
        <span>
          {Object.keys(run.outputs).length} present
          {declaredCount === undefined ? "" : ` · ${declaredCount} declared`}
        </span>
      </div>
      {contract.isPending && identity !== undefined ? (
        <p className="loading-copy">Loading Workflow output contract…</p>
      ) : null}
      {contract.data === undefined ? null : (
        <p className="run-output-intent">
          Primary is the Workflow&apos;s intended display order, not a quality
          mark.
        </p>
      )}
      {contractError === null ? null : (
        <div className="run-output-contract-error">
          <p>
            Output roles are unavailable. Present artifacts remain accessible
            without primary classification.
          </p>
          <ErrorNotice error={contractError} />
        </div>
      )}
      {entries.length === 0 ? (
        <div className="compact-empty">
          {contract.data === undefined
            ? "No Run outputs are currently available."
            : "This Workflow declares no output slots."}
        </div>
      ) : (
        <div className="run-output-list">
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
    <details
      className="panel run-disclosure run-artifact-library"
      id="run-artifacts"
      {...disclosure}
    >
      <RunDisclosureSummary
        eyebrow="Run artifacts"
        title="Inputs, intermediate Artifacts and outputs"
        aside={
          count === undefined
            ? undefined
            : `${count} Artifact${count === "1" ? "" : "s"}${namespace === undefined ? "" : ` in ${namespace}`}`
        }
      />
      <div className="run-disclosure-body">
        {namespaceFilter.form}
        {namespaceFilter.error}
        <QueryView
          query={query}
          loading={
            <p className="loading-copy" role="status">
              Loading Run Artifacts…
            </p>
          }
          onRetry={() => void query.refetch()}
          isEmpty={(page) => page.items.length === 0}
          empty={
            <div className="compact-empty">No bindings match this view.</div>
          }
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
      </div>
    </details>
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
      className="runs-page"
      heading={
        <>
          <ReturnLink to={`/runs/${encodeURIComponent(runId)}`} label="Run" />
          <p className="eyebrow">Run inputs</p>
        </>
      }
    />
  );
}
