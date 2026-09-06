import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  cancelRun,
  getRun,
  isTerminalRunState,
  RUN_ID_PATTERN,
  type RunStatus,
} from "../../api/runs";
import { ErrorNotice, formatTimestamp } from "../artifacts/common";
import {
  DefinitionList,
  RunArtifactRef,
  RunMetadataLabelChips,
  StageAttemptView,
  StateBadge,
} from "./components";
import { RunArtifactLibrary, RunOutputGallery } from "./artifacts";
import { RunResumeControl } from "./resume";
import { useLiveRunProjection } from "./live";
import { deriveRunTriage, formatRunDuration, type RunTriage } from "./triage";

function compactMetric(value: number): string {
  if (value < 1_000) {
    return String(value);
  }
  if (value < 1_000_000) {
    return `${Number((value / 1_000).toFixed(1))}K`;
  }
  return `${Number((value / 1_000_000).toFixed(1))}M`;
}

function triageTitle(run: RunStatus, triage: RunTriage): string {
  const stage = triage.stage === undefined ? "" : ` in ${triage.stage}`;
  switch (run.state) {
    case "failed":
      return `Run failed${stage}`;
    case "succeeded":
      return "Run completed successfully";
    case "cancelled":
      return "Run was cancelled";
    case "cancelling":
      return "Cancellation is in progress";
    case "initializing":
      return "Run is initializing";
    case "running":
      return triage.stage === undefined
        ? "Run is in progress"
        : `Running ${triage.stage}`;
  }
}

function triageDescription(run: RunStatus, triage: RunTriage): string {
  if (triage.issue !== undefined) {
    return triage.issue.message;
  }
  switch (run.state) {
    case "failed":
      return "No failure reason was reported. Inspect the latest attempt for details.";
    case "succeeded":
      return triage.outputCount === 0
        ? "All work completed. This Run has no outputs."
        : `${triage.outputCount} output${triage.outputCount === 1 ? " is" : "s are"} ready to inspect.`;
    case "cancelled":
      return "Work stopped and cleanup completed.";
    case "cancelling":
      return "Stopping active work and cleaning up.";
    case "initializing":
      return "Preparing to start the first stage.";
    case "running":
      return "Follow the current stage and its progress below.";
  }
}

function RunTriageSummary({
  run,
  triage,
}: {
  run: RunStatus;
  triage: RunTriage;
}) {
  const attemptAnchor =
    triage.stageExecutionId === undefined
      ? undefined
      : `#attempt-${triage.stageExecutionId}`;
  return (
    <section
      className={`panel run-triage run-triage-${run.state}`}
      aria-labelledby="run-triage-title"
    >
      <div className="run-triage-heading">
        <div>
          <p className="eyebrow">Run status</p>
          <h3 id="run-triage-title">{triageTitle(run, triage)}</h3>
        </div>
        <StateBadge state={run.state} />
      </div>
      {triage.issue === undefined ? (
        <p className="run-triage-description">
          {triageDescription(run, triage)}
        </p>
      ) : (
        <div className="run-triage-issue">
          <div>
            <span>
              {triage.issue.source === "cancellation"
                ? "Lifecycle reason"
                : "Primary cause"}
            </span>
            <code>{triage.issue.code}</code>
            <strong>
              {triage.issue.source === "cancellation"
                ? "user requested"
                : triage.issue.retryable === undefined
                  ? "retryability unknown"
                  : triage.issue.retryable
                    ? "retryable"
                    : "not retryable"}
            </strong>
          </div>
          <p>{triageDescription(run, triage)}</p>
          {triage.issue.participant === undefined ? null : (
            <small>
              Reported by {triage.issue.participant}
              {triage.issue.logicalAgent === undefined
                ? ""
                : ` ${triage.issue.logicalAgent}`}
            </small>
          )}
        </div>
      )}
      <dl className="run-triage-facts">
        <div>
          <dt>Duration</dt>
          <dd>{formatRunDuration(triage.durationMs)}</dd>
        </div>
        <div>
          <dt>Attempts</dt>
          <dd>{triage.attemptCount}</dd>
          <small>across all stages</small>
        </div>
        <div>
          <dt>Total tokens</dt>
          <dd title={triage.metrics?.totalTokens.toLocaleString()}>
            {triage.metrics === undefined
              ? "—"
              : compactMetric(triage.metrics.totalTokens)}
          </dd>
          <small>
            {triage.metrics === undefined
              ? "not reported"
              : `${triage.metrics.modelCalls} model call${triage.metrics.modelCalls === 1 ? "" : "s"}${triage.metrics.incomplete ? " · partial" : ""}`}
          </small>
        </div>
        <div>
          <dt>Tool calls</dt>
          <dd>
            {triage.metrics === undefined
              ? "—"
              : compactMetric(triage.metrics.toolCalls)}
          </dd>
          <small>
            {triage.metrics === undefined
              ? "not reported"
              : `${triage.metrics.errorCount} reported error${triage.metrics.errorCount === 1 ? "" : "s"}`}
          </small>
        </div>
        <div>
          <dt>Outputs</dt>
          <dd>{triage.outputCount}</dd>
          <small>ready to inspect</small>
        </div>
      </dl>
      {attemptAnchor === undefined && triage.outputCount === 0 ? null : (
        <nav className="run-triage-actions" aria-label="Run triage shortcuts">
          {attemptAnchor === undefined ? null : (
            <a className="triage-action" href={attemptAnchor}>
              Inspect focused attempt
            </a>
          )}
          {triage.outputCount === 0 ? null : (
            <a className="triage-action" href="#run-outputs">
              Preview outputs
            </a>
          )}
        </nav>
      )}
    </section>
  );
}

function CancellationControl({ run }: { run: RunStatus }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [reason, setReason] = useState("");
  const [validationError, setValidationError] = useState<string | undefined>();
  const mutation = useMutation({
    mutationFn: (value: string) => cancelRun(api, run.runId, value),
    onSettled: async () => {
      // Both acceptance and a terminal-completion race are reconciled from the
      // authoritative aggregate. The mutation response is never projected.
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs.all });
    },
  });

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    mutation.reset();
    const normalized = reason.trim();
    if (normalized.length === 0 || normalized.length > 4096) {
      setValidationError("Give a cancellation reason of 1–4096 characters.");
      return;
    }
    setValidationError(undefined);
    mutation.mutate(normalized);
  }

  if (isTerminalRunState(run.state)) {
    return (
      <div className="panel cancellation-panel">
        <p className="eyebrow">Cancellation</p>
        <h3>Run is terminal</h3>
        <p className="muted-copy">
          Cancellation is not available for a terminal Run.
        </p>
      </div>
    );
  }
  if (run.state === "cancelling") {
    return (
      <div className="panel cancellation-panel">
        <p className="eyebrow">Cancellation</p>
        <h3>Cleanup in progress</h3>
        <p className="muted-copy">
          Scheduler is aborting and draining active work. This page does not
          predict when cleanup becomes terminal.
        </p>
      </div>
    );
  }
  return (
    <form className="panel cancellation-panel" onSubmit={submit} noValidate>
      <p className="eyebrow">Bounded lifecycle command</p>
      <h3>Cancel Run</h3>
      <label>
        Explicit reason
        <textarea
          name="cancellationReason"
          maxLength={4096}
          value={reason}
          aria-invalid={validationError === undefined ? undefined : true}
          aria-describedby={
            validationError === undefined ? undefined : "cancel-reason-error"
          }
          onChange={(event) => {
            setReason(event.target.value);
            setValidationError(undefined);
          }}
        />
      </label>
      {validationError === undefined ? null : (
        <p className="field-error" id="cancel-reason-error" role="alert">
          {validationError}
        </p>
      )}
      {mutation.error === null ? null : (
        <>
          <ErrorNotice error={mutation.error} />
          <p className="muted-copy">
            The Run snapshot was refreshed; a concurrent terminal transition
            remains authoritative.
          </p>
        </>
      )}
      <button type="submit" disabled={mutation.isPending}>
        {mutation.isPending
          ? "Requesting cancellation…"
          : "Request cancellation"}
      </button>
      <small>
        No terminal state is applied optimistically; Server cleanup remains
        visible as Stage attempts change.
      </small>
    </form>
  );
}

function RunTimestamps({ run }: { run: RunStatus }) {
  const values: Array<[string, string | undefined]> = [
    ["created", run.createdAt],
    ["updated", run.updatedAt],
    ["started", run.startedAt],
    ["finished", run.finishedAt],
  ];
  return (
    <dl className="metadata-grid panel run-metadata">
      <div>
        <dt>Workflow</dt>
        <dd>
          <code>{run.workflow}</code>
        </dd>
      </div>
      {run.projectId === undefined ? null : (
        <div>
          <dt>Project</dt>
          <dd>
            <Link to={`/projects/${encodeURIComponent(run.projectId)}`}>
              <code>{run.projectId}</code>
            </Link>
          </dd>
        </div>
      )}
      {values.map(([label, value]) =>
        value === undefined ? null : (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{formatTimestamp(value)}</dd>
          </div>
        ),
      )}
    </dl>
  );
}

function RunOutputPublications({ run }: { run: RunStatus }) {
  const projectId = run.projectId;
  if (projectId === undefined) {
    return null;
  }
  return (
    <section className="panel run-output-publications">
      <div className="section-heading">
        <div>
          <p className="eyebrow">ProjectScope publication</p>
          <h3>Reusable output status</h3>
        </div>
        <Link
          to={`/projects/${encodeURIComponent(projectId)}#project-artifacts`}
        >
          Open Project →
        </Link>
      </div>
      {run.outputPublications.length === 0 ? (
        <div className="compact-empty">
          {isTerminalRunState(run.state)
            ? "No present declared output required a publication receipt."
            : "Publication is recorded only after successful terminal output freezing."}
        </div>
      ) : (
        <ul className="run-output-publication-list">
          {run.outputPublications.map((publication) => (
            <li key={`${publication.output}:${publication.source.revision}`}>
              <div>
                <strong>{publication.output}</strong>
                <span
                  className={`publication-status publication-${publication.status}`}
                >
                  {publication.status.replaceAll("_", " ")}
                </span>
              </div>
              <span>
                source{" "}
                <code>
                  {publication.source.namespace}/{publication.source.name}@
                  {publication.source.revision}
                </code>
              </span>
              {publication.target === undefined ? null : (
                <Link
                  to={`/projects/${encodeURIComponent(projectId)}/artifacts/${encodeURIComponent(publication.target.namespace)}/${encodeURIComponent(publication.target.name)}?revision=${encodeURIComponent(publication.target.revision)}`}
                >
                  target {publication.target.namespace}/
                  {publication.target.name}@{publication.target.revision}
                </Link>
              )}
              {publication.errorMessage === undefined ? null : (
                <p>
                  <code>{publication.errorCode ?? "publication_failed"}</code>{" "}
                  {publication.errorMessage}
                </p>
              )}
              <small>{formatTimestamp(publication.createdAt)}</small>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function RunBindings({ run }: { run: RunStatus }) {
  return (
    <div className="run-binding-grid">
      <DefinitionList title="String parameters">
        {Object.entries(run.parameters ?? {}).length === 0 ? (
          <div className="compact-empty">None supplied.</div>
        ) : (
          <dl className="key-value-list">
            {Object.entries(run.parameters ?? {})
              .sort(([left], [right]) => left.localeCompare(right))
              .map(([name, value]) => (
                <div key={name}>
                  <dt>{name}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
          </dl>
        )}
      </DefinitionList>
      <DefinitionList title="Exact input forks">
        {Object.entries(run.inputs ?? {}).length === 0 ? (
          <div className="compact-empty">No inputs.</div>
        ) : (
          <div className="artifact-ref-list">
            {Object.entries(run.inputs ?? {})
              .sort(([left], [right]) => left.localeCompare(right))
              .map(([slot, artifact]) => (
                <RunArtifactRef
                  key={slot}
                  runId={run.runId}
                  slot={slot}
                  artifact={artifact}
                />
              ))}
          </div>
        )}
      </DefinitionList>
    </div>
  );
}

function RunMetadataLabels({ run }: { run: RunStatus }) {
  return (
    <section className="panel run-metadata-label-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Labels</p>
          <h3>Run metadata labels</h3>
        </div>
        <span>{Object.keys(run.labels).length} labels</span>
      </div>
      <p className="muted-copy">
        Use these labels to find related Runs. Labels are fixed at creation.
      </p>
      <RunMetadataLabelChips labels={run.labels} empty="No metadata labels." />
    </section>
  );
}

function RunRuntimeConfiguration({ run }: { run: RunStatus }) {
  const entries = [
    run.runtimeConfiguration.default,
    ...run.runtimeConfiguration.labels,
  ];
  return (
    <details className="panel run-runtime-configuration">
      <summary>
        <span>
          <span className="eyebrow">Pinned at Run creation</span>
          <strong>Runtime infrastructure configuration</strong>
        </span>
        <span>{run.runtimeLabels.length} explicit Runtime labels</span>
      </summary>
      <div className="run-runtime-configuration-body">
        <p className="muted-copy">
          Default and explicit Runtime labels below are immutable for this Run.
          Later Runtime-label rebinding cannot change these exact refs.
        </p>
        <div className="runtime-provenance-grid">
          {entries.map((pin) => (
            <article
              className={
                pin.label === "default" ? "runtime-default-pin" : undefined
              }
              key={`${pin.label}:${pin.bindingRevision}`}
            >
              <strong>
                {pin.label}
                {pin.label === "default" ? " · always applied" : ""}
              </strong>
              <span>binding revision {pin.bindingRevision}</span>
              <code>
                {pin.config.name}@{pin.config.version}
              </code>
              <code title={pin.config.digest}>
                {pin.config.digest.slice(0, 18)}…
              </code>
            </article>
          ))}
        </div>
        <p className="muted-copy">
          Agent-label overrides become knowable only after Scheduler commits an
          allocation snapshot; they are never inferred from current Operations
          state.
        </p>
      </div>
    </details>
  );
}

function LiveAttempts({
  run,
  focusStageExecutionId,
}: {
  run: RunStatus;
  focusStageExecutionId: string | undefined;
}) {
  const live = useLiveRunProjection(run);
  return (
    <>
      <div className={`live-status live-${live.connection}`} role="status">
        <span className="status-dot" aria-hidden="true" />
        Live events: {live.connection}
        {live.resyncReason === undefined
          ? null
          : ` · REST resync after ${live.resyncReason.replaceAll("_", " ")}`}
      </div>
      {live.error === undefined ? null : (
        <div className="notice notice-warning" role="alert">
          <strong>{live.error}</strong>
          <p>Manual refresh remains available and authoritative.</p>
        </div>
      )}
      <section className="run-attempts" id="run-attempts">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Scheduler history</p>
            <h3>Ordered Stage attempts</h3>
          </div>
          <span>
            {run.attempts.length} attempt{run.attempts.length === 1 ? "" : "s"}
          </span>
        </div>
        {run.attempts.length === 0 ? (
          <div className="panel compact-empty">
            No Stage attempt has been durably created yet.
          </div>
        ) : (
          run.attempts.map((attempt) => (
            <StageAttemptView
              key={attempt.stageExecutionId}
              runId={run.runId}
              attempt={attempt}
              active={run.activeStageExecutionId === attempt.stageExecutionId}
              focused={focusStageExecutionId === attempt.stageExecutionId}
              projection={live.planners[attempt.stageExecutionId] ?? {}}
              transitions={run.transitions.filter(
                (transition) =>
                  transition.sourceExecutionId === attempt.stageExecutionId,
              )}
            />
          ))
        )}
      </section>
    </>
  );
}

function LoadedRunDetail({
  run,
  snapshotVersion,
}: {
  run: RunStatus;
  snapshotVersion: number;
}) {
  const queryClient = useQueryClient();
  const invalidatedPublication = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (run.projectId === undefined || run.state !== "succeeded") {
      return;
    }
    const identity = `${run.runId}:${run.finishedAt ?? run.updatedAt ?? "terminal"}`;
    if (invalidatedPublication.current === identity) {
      return;
    }
    invalidatedPublication.current = identity;
    void queryClient.invalidateQueries({
      queryKey: queryKeys.projects.artifacts.all(run.projectId),
    });
  }, [
    queryClient,
    run.finishedAt,
    run.projectId,
    run.runId,
    run.state,
    run.updatedAt,
  ]);
  const liveKey = `${run.eventCursor?.generation ?? "none"}:${run.eventCursor?.sequence ?? "none"}`;
  const triage = deriveRunTriage(run);
  return (
    <>
      <RunTriageSummary run={run} triage={triage} />
      <RunResumeControl
        key={`${run.runId}:${run.resumeStageExecutionId ?? "none"}`}
        run={run}
      />
      {run.cancellation === undefined ? null : (
        <div className="notice notice-warning cancellation-record">
          <strong>Cancellation requested</strong>
          <span>
            <code>{run.cancellation.code}</code> ·{" "}
            {formatTimestamp(run.cancellation.requestedAt)}
          </span>
          {run.cancellation.reason === undefined ? null : (
            <p>{run.cancellation.reason}</p>
          )}
          {run.cancellation.requestedBy === undefined ? null : (
            <small>Requested by {run.cancellation.requestedBy}</small>
          )}
        </div>
      )}
      <RunOutputGallery runId={run.runId} outputs={run.outputs} />
      <RunOutputPublications run={run} />
      <LiveAttempts
        key={`${liveKey}:${snapshotVersion}`}
        run={run}
        focusStageExecutionId={triage.stageExecutionId}
      />

      <RunMetadataLabels run={run} />
      <RunTimestamps run={run} />
      <RunBindings run={run} />

      <RunRuntimeConfiguration run={run} />

      <RunArtifactLibrary runId={run.runId} />
      <CancellationControl run={run} />
    </>
  );
}

export function RunDetailRoute() {
  const api = usePublicAPI();
  const { runId = "" } = useParams();
  const [copiedId, setCopiedId] = useState<string>();
  const [copyError, setCopyError] = useState<string>();
  const valid = RUN_ID_PATTERN.test(runId);
  const query = useQuery({
    queryKey: queryKeys.runs.detail(runId),
    queryFn: () => getRun(api, runId),
    enabled: valid,
  });

  async function copyRunId(): Promise<void> {
    setCopyError(undefined);
    try {
      await navigator.clipboard.writeText(runId);
      setCopiedId(runId);
    } catch {
      setCopyError("Could not copy. Select the Run ID to copy it manually.");
    }
  }
  if (!valid) {
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Run route is invalid")} />
        <Link to="/runs">Return to Runs</Link>
      </section>
    );
  }
  return (
    <section className="route-page runs-page run-detail-page">
      <header className="route-header-row">
        <div>
          <Link className="back-link" to="/runs">
            ← All Runs
          </Link>
          <p className="eyebrow">Workflow Run</p>
          <h2>{query.data?.workflow ?? "Run details"}</h2>
          <div className="run-identity">
            <code title={runId}>{runId}</code>
            <button
              className="secondary-button"
              type="button"
              aria-label="Copy Run ID"
              onClick={() => void copyRunId()}
            >
              {copiedId === runId ? "Copied" : "Copy ID"}
            </button>
            <span className="visually-hidden" role="status">
              {copiedId === runId ? "Run ID copied" : ""}
            </span>
          </div>
          {copyError === undefined ? null : (
            <p className="field-error" role="alert">
              {copyError}
            </p>
          )}
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
          Loading Run…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : (
        <LoadedRunDetail
          key={query.data.runId}
          run={query.data}
          snapshotVersion={query.dataUpdatedAt}
        />
      )}
    </section>
  );
}
