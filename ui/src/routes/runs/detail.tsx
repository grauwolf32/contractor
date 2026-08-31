import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
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
  StageAttemptView,
  StateBadge,
} from "./components";
import { RunArtifactLibrary } from "./artifacts";
import { useLiveRunProjection } from "./live";

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
          No lifecycle command is available for a terminal Run.
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
      <div>
        <dt>State</dt>
        <dd>
          <StateBadge state={run.state} />
        </dd>
      </div>
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
      <DefinitionList title="Frozen Workflow outputs">
        {Object.entries(run.outputs).length === 0 ? (
          <div className="compact-empty">No frozen outputs yet.</div>
        ) : (
          <div className="artifact-ref-list">
            {Object.entries(run.outputs)
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

function LiveAttempts({ run }: { run: RunStatus }) {
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
      <section className="run-attempts">
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
  const liveKey = `${run.eventCursor?.generation ?? "none"}:${run.eventCursor?.sequence ?? "none"}`;
  return (
    <>
      <RunTimestamps run={run} />
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
      <RunBindings run={run} />
      <LiveAttempts key={`${liveKey}:${snapshotVersion}`} run={run} />

      <RunArtifactLibrary runId={run.runId} />
      <CancellationControl run={run} />
    </>
  );
}

export function RunDetailRoute() {
  const api = usePublicAPI();
  const { runId = "" } = useParams();
  const valid = RUN_ID_PATTERN.test(runId);
  const query = useQuery({
    queryKey: queryKeys.runs.detail(runId),
    queryFn: () => getRun(api, runId),
    enabled: valid,
  });
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
          <p className="eyebrow">Authoritative aggregate</p>
          <h2>{runId}</h2>
          <p className="lede">
            Stage lifecycle, Scheduler decisions, exact Artifacts and aggregate
            metrics come from REST. Typed Planner facts update only the nested
            live plan between snapshots.
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
          Loading Run aggregate…
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
