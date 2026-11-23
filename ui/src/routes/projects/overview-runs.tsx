import { type UseQueryResult, useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { getRun, type RunPage, type RunSummary } from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import {
  Button,
  IdChip,
  IconButton,
  StatusChip,
  TechnicalDetails,
} from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import {
  organizeRunOutputs,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
} from "../runs/output-model";
import { ArtifactPreviewPanel } from "../artifacts/preview";
import { useRunArtifactActions } from "../artifacts/use-run-artifact-actions";
import type { OutputEntry } from "../runs/output-model";
import { workflowFormats } from "../workflows/formats";
import { projectPath } from "./project-sections";
import { runStateLabel } from "../../app/vocabulary";
import { Icon } from "../../app/icon";
import { RunListSourceContext } from "../runs/source-context";
import { RunSourceLink } from "../runs/source-link";
import { OVERVIEW_POLL_MS, SUCCESSFUL_RUNS_POLL_MS } from "./overview-data";

/** A Run's exact outputs and its Workflow's published output roles. */
function useRunOutputs(summary: RunSummary) {
  const api = usePublicAPI();
  const identity = parseWorkflowIdentity(summary.workflow);
  const run = useQuery({
    queryKey: queryKeys.runs.detail(summary.runId),
    queryFn: () => getRun(api, summary.runId),
  });
  const contract = useQuery({
    queryKey: [
      ...queryKeys.workflows.detail(
        identity?.name ?? "",
        identity?.version ?? "",
      ),
      "outputs",
    ],
    queryFn: async () => {
      if (!identity) throw new Error("Workflow identity is invalid");
      return requireWorkflowOutputs(
        await getWorkflow(api, identity.name, identity.version),
        identity,
      );
    },
    enabled: identity !== undefined,
  });
  return { run, contract };
}

/**
 * The published primary results of a succeeded Run, as links: nothing while
 * the roles are unknown or when no primary output was published, so no other
 * file is ever offered in its place (US-05).
 */
export function RunPrimaryResult({
  summary,
  projectId,
}: {
  summary: RunSummary;
  projectId: string;
}) {
  const { run, contract } = useRunOutputs(summary);
  if (run.data === undefined || run.data.projectId !== projectId) return null;
  const primary = organizeRunOutputs(run.data.outputs, contract.data).flatMap(
    ({ kind, slot, artifact }) =>
      kind === "primary" && artifact !== undefined ? [{ slot, artifact }] : [],
  );
  if (primary.length === 0) return null;
  return (
    <>
      {" "}
      Primary result:{" "}
      {primary.map(({ slot, artifact }, position) => (
        <span key={slot}>
          {position > 0 ? ", " : null}
          <ContextLink
            returnLabel="Project Overview"
            to={artifactDetailPath(
              { kind: "run", id: summary.runId },
              artifact,
            )}
          >
            {slot}
          </ContextLink>
        </span>
      ))}
      .
    </>
  );
}

/** The producing identity stays secondary to the document itself. */
function ResultProvenance({ summary }: { summary: RunSummary }) {
  return (
    <div className="projects-result-provenance">
      <RunSourceLink
        returnLabel="Project Overview"
        to={`/runs/${encodeURIComponent(summary.runId)}`}
      >
        {summary.workflow}
      </RunSourceLink>
      <IdChip value={summary.runId} label="Run ID" />
      <span className="projects-run-time">
        {summary.finishedAt ? "Finished" : "Updated"}{" "}
        <RecordedTime
          display="clock"
          value={summary.finishedAt ?? summary.updatedAt}
        />
      </span>
    </div>
  );
}

function ProjectResult({
  summary,
  entry,
}: {
  summary: RunSummary;
  entry: OutputEntry;
}) {
  const titleId = useId();
  const { requested, requestPreview, metadata, download, loadPreview } =
    useRunArtifactActions(summary.runId, entry.artifact);
  const artifact = entry.artifact;
  if (!artifact) return null;
  const exactRef = `${artifact.namespace}/${artifact.name}@${artifact.revision}`;
  return (
    <article className="projects-output" aria-labelledby={titleId}>
      <div className="projects-output-head">
        <span className="projects-output-icon">
          <Icon name="document-code" />
        </span>
        <div className="projects-output-text">
          <h4 id={titleId}>{entry.slot}</h4>
          <small>
            {entry.kind === "primary"
              ? "Primary result"
              : entry.kind === "declared"
                ? "Supporting result"
                : "Saved result"}
            {entry.declaration
              ? ` · ${entry.declaration.mediaTypes.map((type) => workflowFormats[type] ?? type).join(" / ")}`
              : ""}
          </small>
        </div>
        <div className="projects-output-actions">
          {requested ? null : (
            <Button size="sm" onClick={requestPreview}>
              Preview
            </Button>
          )}
          <IconButton
            size="sm"
            label={
              download.isPending
                ? `Downloading ${entry.slot}…`
                : `Download ${entry.slot}`
            }
            pending={download.isPending}
            icon={<Icon name="download" />}
            onClick={() => download.mutate()}
          />
        </div>
      </div>
      <div className="projects-output-ref">
        <IdChip value={exactRef} label={`${entry.slot} revision`} />
      </div>
      {download.error === null ? null : (
        <ErrorNotice
          error={download.error}
          context={`Could not download ${entry.slot}`}
        />
      )}
      {requested ? (
        <QueryView
          query={metadata}
          onRetry={() => void metadata.refetch()}
          loading={<p className="ui-caption">Loading output metadata…</p>}
        >
          {(data) => (
            <ArtifactPreviewPanel
              archiveScope={{ kind: "run", id: summary.runId }}
              key={data.artifact.revision}
              metadata={data}
              unavailableCopy="Inline preview is unavailable for this file. Use Download for the original bytes, or open its details."
              loadPreview={() => loadPreview(data)}
              loadOnMountKey={[
                "run-output-preview",
                summary.runId,
                data.artifact.namespace,
                data.artifact.name,
                data.artifact.revision,
              ]}
            />
          )}
        </QueryView>
      ) : null}
      <footer className="projects-output-footer">
        <ResultProvenance summary={summary} />
        <ContextLink
          returnLabel="Project Overview"
          to={artifactDetailPath({ kind: "run", id: summary.runId }, artifact)}
          aria-label={`${entry.slot} details`}
        >
          details →
        </ContextLink>
      </footer>
    </article>
  );
}

/** Declared primary results first; never substitute another file for a missing one. */
function RecentRunResult({
  summary,
  projectId,
}: {
  summary: RunSummary;
  projectId: string;
}) {
  const { run, contract } = useRunOutputs(summary);
  const entries = organizeRunOutputs(run.data?.outputs ?? {}, contract.data);
  const primary = entries.filter((entry) => entry.kind === "primary");
  const shown = (primary.length ? primary : entries).slice(0, 2);
  return (
    <li className="projects-result">
      {run.isPending ? (
        <p className="loading-copy" role="status">
          Loading outputs…
        </p>
      ) : run.error ? (
        <ErrorNotice
          error={run.error}
          onRetry={() => void run.refetch()}
          retryLabel="Retry outputs"
          retryPending={run.isFetching}
        />
      ) : run.data.projectId !== projectId ? (
        <p className="form-error">Run does not belong to this project.</p>
      ) : (
        <>
          {contract.error ? (
            <TechnicalDetails
              summary={
                contract.data === undefined
                  ? "Result labels could not be loaded"
                  : "Result labels could not be refreshed"
              }
            >
              <p className="ui-caption">
                {contract.data === undefined
                  ? "Saved files are still available. The primary and supporting labels are unknown until their Workflow is loaded."
                  : "Showing the previously loaded primary and supporting labels."}
              </p>
              <p className="ui-caption">{contract.error.message}</p>
              <Button size="xs" onClick={() => void contract.refetch()}>
                Retry result labels
              </Button>
            </TechnicalDetails>
          ) : null}
          {shown.length === 0 ? (
            <>
              <p className="ui-caption">No output published by this Run.</p>
              <ResultProvenance summary={summary} />
            </>
          ) : (
            <ul role="list" className="projects-outputs">
              {shown.map((entry) => (
                <li
                  key={[
                    entry.slot,
                    entry.artifact?.namespace,
                    entry.artifact?.name,
                    entry.artifact?.revision,
                  ].join(":")}
                >
                  {entry.artifact ? (
                    <ProjectResult entry={entry} summary={summary} />
                  ) : (
                    <div className="projects-output">
                      <p className="ui-caption">
                        {entry.kind === "primary" ? "Primary result" : "Output"}{" "}
                        <code>{entry.slot}</code> was not published.
                      </p>
                      <ResultProvenance summary={summary} />
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </li>
  );
}

/** Outputs of the three most recent successful Runs. */
export function RecentResults({
  projectId,
  results,
}: {
  projectId: string;
  results: UseQueryResult<RunPage>;
}) {
  const heading = useId();
  return (
    <RunListSourceContext
      value={{
        kind: "project",
        label: "Successful Runs",
        projectId,
        options: { limit: 3, state: "succeeded" },
        refreshMs: SUCCESSFUL_RUNS_POLL_MS,
      }}
    >
      <section className="projects-recent-results" aria-labelledby={heading}>
        <div className="projects-section-heading">
          <h3 id={heading}>Recent results</h3>
          <Link
            to={`${projectPath(projectId, "runs")}?view=completed&state=succeeded`}
          >
            Successful Runs
          </Link>
        </div>
        <p className="ui-caption">
          Outputs of the three most recent successful Runs.
        </p>
        <QueryView
          query={results}
          loading={
            <p className="loading-copy" role="status">
              Loading successful Runs…
            </p>
          }
          onRetry={() => void results.refetch()}
          isEmpty={(data) => data.items.length === 0}
          empty={
            <p className="ui-caption">
              No successful Runs to show. Published results appear here.
            </p>
          }
        >
          {(data) => (
            <ul role="list" className="projects-results">
              {data.items.map((run) => (
                <RecentRunResult
                  key={run.runId}
                  summary={run}
                  projectId={projectId}
                />
              ))}
            </ul>
          )}
        </QueryView>
      </section>
    </RunListSourceContext>
  );
}

/** The five most recent Runs of the project, in any state. */
export function RecentRuns({
  projectId,
  runs,
}: {
  projectId: string;
  runs: UseQueryResult<RunPage>;
}) {
  const heading = useId();
  return (
    <RunListSourceContext
      value={{
        kind: "project",
        label: "Recent Runs",
        projectId,
        options: { limit: 5 },
        refreshMs: OVERVIEW_POLL_MS,
      }}
    >
      <section className="projects-recent-runs" aria-labelledby={heading}>
        <div className="projects-section-heading">
          <h3 id={heading}>Recent Runs</h3>
          <Link to={projectPath(projectId, "runs")}>All Runs</Link>
        </div>
        <QueryView
          query={runs}
          loading={
            <p className="loading-copy" role="status">
              Loading recent Runs…
            </p>
          }
          onRetry={() => void runs.refetch()}
          isEmpty={(data) => data.items.length === 0}
          empty={
            <p className="ui-caption">
              No Runs to show. Checks and Workflows start Runs.
            </p>
          }
        >
          {(data) => (
            <ul role="list" className="projects-recent-run-list">
              {data.items.map((run) => {
                const state = runStateLabel(run.state);
                return (
                  <li className="projects-recent-run" key={run.runId}>
                    <div className="projects-recent-run-top">
                      <RunSourceLink
                        className="projects-recent-run-link"
                        returnLabel="Project Overview"
                        to={`/runs/${encodeURIComponent(run.runId)}`}
                      >
                        <strong>
                          {parseWorkflowIdentity(run.workflow)?.name ??
                            run.workflow}
                        </strong>
                        {parseWorkflowIdentity(run.workflow) ? (
                          <small>
                            @{parseWorkflowIdentity(run.workflow)?.version}
                          </small>
                        ) : null}
                      </RunSourceLink>
                      <StatusChip tone={state.tone} size="sm">
                        {state.label}
                      </StatusChip>
                    </div>
                    <div className="projects-result-provenance">
                      <IdChip value={run.runId} label="Run ID" />
                      <span className="projects-run-time">
                        {run.finishedAt ? "Finished" : "Updated"}{" "}
                        <RecordedTime
                          display="clock"
                          value={run.finishedAt ?? run.updatedAt}
                        />
                      </span>
                      <Icon name="chevron-right" size={14} />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </QueryView>
      </section>
    </RunListSourceContext>
  );
}
