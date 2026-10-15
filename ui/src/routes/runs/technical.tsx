import { useEffect } from "react";
import { Link } from "react-router";

import type { RunStatus } from "../../api/runs";
import { runtimeConfigVersionPath } from "../../app/navigation";
import { formatElapsed, formatTimestamp, plural } from "../../app/format";
import { useSession } from "../../auth/session";
import { Facts } from "../../ui";
import { StageAttemptView } from "./attempt";
import {
  DefinitionList,
  Digest,
  RunArtifactRef,
  type RunDisclosureProps,
  RunMetadataLabelChips,
  RunSection,
} from "./components";
import { type LiveRunProjection, useLiveRunProjection } from "./live";
import type { RunTriage } from "./triage";

/*
 * The Run page's Technical details sections, each a collapsible RunSection
 * whose open state the page keeps.
 */

/** The event stream's state, as the page header shows it. */
export type LiveStatus = Pick<
  LiveRunProjection,
  "connection" | "error" | "resyncReason"
>;

// V385: needs shared compact number formatter ("1.2K", "3.4M") in
// src/app/format.ts.
function compactMetric(value: number): string {
  if (value < 1_000) {
    return String(value);
  }
  if (value < 1_000_000) {
    return `${Number((value / 1_000).toFixed(1))}K`;
  }
  return `${Number((value / 1_000_000).toFixed(1))}M`;
}

export function RunExecutionMetrics({
  run,
  triage,
  ...disclosure
}: { run: RunStatus; triage: RunTriage } & RunDisclosureProps) {
  return (
    <RunSection
      id="run-metrics"
      title="Execution metrics"
      description="Duration, attempts, tokens and tool calls"
      aside={`${formatElapsed(triage.durationMs)} · ${plural(triage.attemptCount, "attempt")}`}
      {...disclosure}
    >
      <dl className="runs-metric-tiles">
        <div>
          <dt>Duration</dt>
          <dd>{formatElapsed(triage.durationMs)}</dd>
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
              : `${plural(triage.metrics.modelCalls, "model call")}${triage.metrics.incomplete ? " · partial" : ""}`}
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
              : plural(triage.metrics.errorCount, "reported error")}
          </small>
        </div>
        <div>
          <dt>Outputs</dt>
          <dd>{Object.keys(run.outputs).length}</dd>
          <small>ready to inspect</small>
        </div>
      </dl>
    </RunSection>
  );
}

export function RunBindings({
  run,
  ...disclosure
}: { run: RunStatus } & RunDisclosureProps) {
  const parameters = Object.entries(run.parameters ?? {}).sort(
    ([left], [right]) => left.localeCompare(right),
  );
  const inputs = Object.entries(run.inputs ?? {}).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  return (
    <RunSection
      title="Parameters and input revisions"
      description="What this Run was started with"
      aside={`${plural(parameters.length, "parameter")} · ${plural(inputs.length, "input")}`}
      {...disclosure}
    >
      <div className="runs-binding-grid">
        <DefinitionList title="String parameters">
          {parameters.length === 0 ? (
            <p className="ui-caption">None supplied.</p>
          ) : (
            <Facts
              className="runs-facts-list"
              items={parameters.map(([name, value]) => ({
                term: name,
                value,
              }))}
            />
          )}
        </DefinitionList>
        <DefinitionList title="Input revisions">
          {inputs.length === 0 ? (
            <p className="ui-caption">No inputs.</p>
          ) : (
            <div className="runs-ref-list">
              {inputs.map(([slot, artifact]) => (
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
    </RunSection>
  );
}

export function RunMetadataLabels({
  run,
  ...disclosure
}: { run: RunStatus } & RunDisclosureProps) {
  return (
    <RunSection
      className="run-metadata-label-panel"
      title="Run metadata labels"
      description="Labels are fixed at creation"
      aside={plural(Object.keys(run.labels).length, "label")}
      {...disclosure}
    >
      <p className="ui-caption">
        Use these labels to find related Runs. Labels are fixed at creation.
      </p>
      <RunMetadataLabelChips labels={run.labels} empty="No metadata labels." />
    </RunSection>
  );
}

export function RunRuntimeConfiguration({
  run,
  ...disclosure
}: {
  run: RunStatus;
} & RunDisclosureProps) {
  const entries = [
    run.runtimeConfiguration.default,
    ...run.runtimeConfiguration.labels,
  ];
  const { session } = useSession();
  const operator =
    session?.principal.capabilities.includes("operations") === true;
  return (
    <RunSection
      className="run-runtime-configuration"
      title="Runtime infrastructure configuration"
      description="Pinned at Run creation"
      aside={plural(run.runtimeLabels.length, "explicit Runtime label")}
      {...disclosure}
    >
      <p className="ui-caption">
        Runtime labels below are pinned for this Run and cannot be rebound
        later.
      </p>
      <div className="runs-pin-grid">
        {entries.map((pin) => (
          <article
            className="runs-pin"
            data-default={pin.label === "default" ? "" : undefined}
            key={`${pin.label}:${pin.bindingRevision}`}
          >
            <strong>
              {pin.label}
              {pin.label === "default" ? " · always applied" : ""}
            </strong>
            <span>binding revision {pin.bindingRevision}</span>
            {operator ? (
              <Link to={runtimeConfigVersionPath(pin.config)}>
                <code>
                  {pin.config.name}@{pin.config.version}
                </code>
              </Link>
            ) : (
              <code>
                {pin.config.name}@{pin.config.version}
              </code>
            )}
            <Digest value={pin.config.digest} />
          </article>
        ))}
      </div>
      <p className="ui-caption">
        Agent-label overrides become knowable only after Scheduler commits an
        allocation snapshot; they are never inferred from current Operations
        state.
      </p>
    </RunSection>
  );
}

export function RunCancellationRecord({
  cancellation,
  ...disclosure
}: {
  cancellation: NonNullable<RunStatus["cancellation"]>;
} & RunDisclosureProps) {
  return (
    <RunSection
      title="Cancellation record"
      description="Who asked to stop this Run, when and why"
      aside={formatTimestamp(cancellation.requestedAt)}
      {...disclosure}
    >
      <Facts
        className="runs-facts-list"
        items={[
          { term: "Code", value: <code>{cancellation.code}</code> },
          {
            term: "Requested",
            value: formatTimestamp(cancellation.requestedAt),
          },
          { term: "Requested by", value: cancellation.requestedBy },
          { term: "Reason", value: cancellation.reason },
        ]}
      />
    </RunSection>
  );
}

/**
 * The ordered Stage attempts with their live Planner projections. It reports
 * the event stream's state to the page header.
 */
export function LiveAttempts({
  run,
  focusStageExecutionId,
  attemptDisclosure,
  instructionDisclosure,
  onLiveStatus,
}: {
  run: RunStatus;
  focusStageExecutionId: string | undefined;
  attemptDisclosure: (
    stageExecutionId: string,
    defaultOpen: boolean,
  ) => RunDisclosureProps;
  instructionDisclosure: (
    stageExecutionId: string,
    subtaskId: string,
  ) => RunDisclosureProps;
  onLiveStatus: (status: LiveStatus) => void;
}) {
  const live = useLiveRunProjection(run);
  useEffect(() => {
    onLiveStatus({
      connection: live.connection,
      ...(live.error === undefined ? {} : { error: live.error }),
      ...(live.resyncReason === undefined
        ? {}
        : { resyncReason: live.resyncReason }),
    });
  }, [live.connection, live.error, live.resyncReason, onLiveStatus]);
  return (
    <section className="runs-attempts" aria-label="Ordered Stage attempts">
      {run.attempts.length === 0 ? (
        <p className="ui-caption">
          No Stage attempt has been durably created yet.
        </p>
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
            disclosure={attemptDisclosure(
              attempt.stageExecutionId,
              run.activeStageExecutionId === attempt.stageExecutionId ||
                focusStageExecutionId === attempt.stageExecutionId,
            )}
            instructionDisclosure={(subtaskId) =>
              instructionDisclosure(attempt.stageExecutionId, subtaskId)
            }
          />
        ))
      )}
    </section>
  );
}
