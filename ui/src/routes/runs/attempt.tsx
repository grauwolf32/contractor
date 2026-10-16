import { lazy, Suspense } from "react";

import type { components } from "../../api/generated/public";
import type { StageAttempt, StageTransition } from "../../api/runs";
import { compactId, formatTimestamp } from "../../app/format";
import { Facts, IdChip, StatusChip, StatusGlyph } from "../../ui";
import { AllocationResourceList } from "../operations/performance/resources";
import { ConsumerConfigList } from "../workflows/consumer-config";
import {
  Digest,
  DisclosureChevron,
  RunArtifactRef,
  type RunDisclosureProps,
} from "./components";
import type { PlannerProjection } from "./planner-projection";
import {
  allocationStatusLabel,
  stageStateLabel,
  subtaskStatusLabel,
} from "./run-state";

type Metrics = components["schemas"]["MetricsSummary"];
type Diagnostics = components["schemas"]["AttemptDiagnostics"];

const MarkdownPreview = lazy(() => import("../artifacts/previews/markdown"));

/** Planner summaries are Markdown; the preview is bounded and lazy-loaded. */
function ResultSummary({ summary }: { summary: string }) {
  if (summary.trim() === "") {
    return null;
  }
  return (
    <div className="result-summary">
      <Suspense fallback={<p className="result-summary-plain">{summary}</p>}>
        <MarkdownPreview source={summary} />
      </Suspense>
    </div>
  );
}

/**
 * A stage execution ID in full, with a copy button: Technical details are
 * where people match attempts and search logs, so the exact value stays
 * readable and copyable (the attempt summary shows only a compact form). On
 * a phone it wraps instead of ending in an ellipsis.
 */
function ExecutionId({ value, label }: { value: string; label: string }) {
  return <IdChip value={value} display={value} label={label} wrap />;
}

function ExecutionConfigView({ attempt }: { attempt: StageAttempt }) {
  const config = attempt.executionConfig;
  return (
    <div className="runs-block">
      <h4 className="runs-block-title">Resolved execution configuration</h4>
      <p className="runs-block-note">
        Variant <code>{config.variant}</code>
        {config.escalationOrdinal === undefined
          ? null
          : ` · escalation ${config.escalationOrdinal}`}
      </p>
      {config.ref === undefined ? null : (
        <p className="runs-block-note">
          Profile{" "}
          <code>
            {config.ref.configId}@{config.ref.version}
          </code>{" "}
          <Digest value={config.ref.digest} />
        </p>
      )}
      <ConsumerConfigList
        config={config}
        workerLabel={(name) => `Logical Worker ${name}`}
        exact
        className="runs-consumer-list"
      />
    </div>
  );
}

function RuntimeConfigurationView({ attempt }: { attempt: StageAttempt }) {
  const configuration = attempt.runtimeConfiguration;
  if (configuration === undefined) {
    return null;
  }
  return (
    <div className="runs-block runs-block-wide">
      <h4 className="runs-block-title">
        Allocation-pinned Runtime configuration
      </h4>
      <p className="runs-block-note">
        This historical projection appears only after placement commits. It
        contains safe refs and origins, never RuntimeSettings or physical Agent
        identity.
      </p>
      <div className="runs-allocation-list">
        {configuration.allocations.map((allocation) => {
          const status = allocationStatusLabel(allocation.status);
          const origins = Object.entries(allocation.origins);
          return (
            <article className="runs-allocation" key={allocation.logicalAgent}>
              <div className="runs-allocation-head">
                <strong>Logical Worker {allocation.logicalAgent}</strong>
                <StatusChip tone={status.tone} size="sm">
                  {status.label}
                </StatusChip>
              </div>
              <Facts
                className="runs-facts-list"
                items={[
                  {
                    term: "Final Agent labels",
                    value:
                      allocation.agentLabels.length === 0
                        ? "none"
                        : allocation.agentLabels.map((pin) => (
                            <span className="runs-agent-pin" key={pin.label}>
                              <strong>{pin.label}</strong>
                              <code>
                                {pin.config.name}@{pin.config.version}
                              </code>
                              <small>
                                binding revision {pin.bindingRevision}
                              </small>
                              <Digest value={pin.config.digest} />
                            </span>
                          )),
                  },
                  {
                    term: "Required adapters",
                    value:
                      allocation.runtimeAdapters.length === 0
                        ? "none"
                        : allocation.runtimeAdapters.map((adapter) => (
                            <code key={adapter}>{adapter}</code>
                          )),
                  },
                  {
                    term: "Final safe origins",
                    value:
                      origins.length === 0 ? (
                        "none"
                      ) : (
                        <ul className="runs-origin-list">
                          {origins.map(([field, origin]) => (
                            <li
                              className={
                                origin.layer === "agent_labels"
                                  ? "runtime-agent-override"
                                  : undefined
                              }
                              key={field}
                            >
                              <strong>{field}</strong>: {origin.layer}
                              {origin.configs?.map((ref) => (
                                <code key={`${ref.name}@${ref.version}`}>
                                  {ref.name}@{ref.version}
                                </code>
                              ))}
                            </li>
                          ))}
                        </ul>
                      ),
                  },
                ]}
              />
            </article>
          );
        })}
      </div>
    </div>
  );
}

function PlannerPlanView({
  projection,
  instructionDisclosure,
}: {
  projection: PlannerProjection;
  instructionDisclosure: (subtaskId: string) => RunDisclosureProps;
}) {
  const plan = projection.plan;
  if (plan === undefined) {
    return (
      <div className="runs-block">
        <h4 className="runs-block-title">Subtasks</h4>
        <p className="runs-block-note">
          No subtask plan is present for this attempt.
        </p>
      </div>
    );
  }
  const dispatch = projection.dispatch ?? plan.activeDispatch;
  const dispatchPhase = projection.dispatch?.phase ?? "running";
  return (
    <div className="runs-block runs-block-wide">
      <div className="runs-block-head">
        <h4 className="runs-block-title">Subtasks</h4>
        <span className="runs-block-aside">
          Nested Planner work · revision {plan.revision}
        </span>
      </div>
      {dispatch === undefined ? null : (
        <div className="runs-dispatch">
          <StatusGlyph tone="progress" />
          <span>
            <strong>Logical Worker {dispatch.workerName}</strong>
            <span>
              subtask {dispatch.subtaskId} · {dispatchPhase}
            </span>
            <code>{dispatch.callId}</code>
          </span>
        </div>
      )}
      <ol className="runs-subtask-list">
        {plan.subtasks.map((subtask) => {
          const current = plan.currentSubtaskId === subtask.id;
          const status = subtaskStatusLabel(subtask.status);
          return (
            <li data-current={current ? "" : undefined} key={subtask.id}>
              <div className="runs-subtask-head">
                <code>{subtask.id}</code>
                <StatusChip tone={status.tone} size="sm">
                  {status.label}
                </StatusChip>
                {current ? (
                  <strong className="runs-marker">current</strong>
                ) : null}
              </div>
              <p>{subtask.objective}</p>
              <details {...instructionDisclosure(subtask.id)}>
                <summary>Planner instructions</summary>
                <p>{subtask.instructions}</p>
              </details>
            </li>
          );
        })}
      </ol>
      {projection.lastEventKind === undefined ? null : (
        <small className="runs-block-note">
          Last event: {projection.lastEventKind}
          {projection.lastOccurredAt === undefined
            ? ""
            : ` · ${formatTimestamp(projection.lastOccurredAt)}`}
        </small>
      )}
    </div>
  );
}

function MetricsView({ metrics }: { metrics: Metrics }) {
  return (
    <div className="runs-block">
      <h4 className="runs-block-title">Safe aggregate metrics</h4>
      <Facts
        className="runs-metrics"
        items={[
          { term: "Model calls", value: metrics.modelCalls },
          { term: "Input tokens", value: metrics.inputTokens },
          { term: "Output tokens", value: metrics.outputTokens },
          { term: "Total tokens", value: metrics.totalTokens },
          { term: "Tool calls", value: metrics.toolCalls },
          { term: "Tool failures", value: metrics.toolFailures },
          { term: "Errors", value: metrics.errorCount },
        ]}
      />
      <p className="runs-block-note">
        Reports {metrics.reportsComplete ? "complete" : "incomplete"}
        {metrics.truncated ? " · bounded data truncated" : ""}
      </p>
    </div>
  );
}

function ResourceView({ attempt }: { attempt: StageAttempt }) {
  if (attempt.resources === undefined || attempt.resources.length === 0) {
    return null;
  }
  return (
    <div className="runs-block runs-block-wide">
      <h4 className="runs-block-title">Allocation resource observations</h4>
      <p className="runs-block-note">
        These terminal summaries use the allocation-pinned collection policy;
        missing measurements are not zero usage.
      </p>
      <AllocationResourceList items={attempt.resources} showRunLink={false} />
    </div>
  );
}

function retryability(retryable: boolean | undefined): string {
  return retryable === undefined
    ? "retryability unknown"
    : retryable
      ? "retryable"
      : "not retryable";
}

function DiagnosticsView({ diagnostics }: { diagnostics: Diagnostics }) {
  return (
    <div className="runs-block">
      <h4 className="runs-block-title">Attempt diagnostics</h4>
      {diagnostics.items.length === 0 ? (
        <p className="runs-block-note">
          No normalized Planner or Worker errors were reported.
        </p>
      ) : (
        <ol className="runs-record-list">
          {diagnostics.items.map((diagnostic, index) => (
            <li
              key={`${diagnostic.participant}-${diagnostic.logicalAgent ?? "planner"}-${diagnostic.code}-${index}`}
            >
              <div className="runs-record-head">
                <strong>
                  {diagnostic.participant === "planner" ? "Planner" : "Worker"}
                </strong>
                {diagnostic.logicalAgent === undefined ? null : (
                  <code>{diagnostic.logicalAgent}</code>
                )}
                <code>{diagnostic.code}</code>
                <span>{retryability(diagnostic.retryable)}</span>
              </div>
              <p>{diagnostic.message}</p>
            </li>
          ))}
        </ol>
      )}
      {diagnostics.truncated ? (
        <p className="runs-block-note">
          Older diagnostics were omitted by a bounded report or public response.
        </p>
      ) : null}
    </div>
  );
}

function ResultView({
  runId,
  attempt,
}: {
  runId: string;
  attempt: StageAttempt;
}) {
  if (attempt.result === undefined && attempt.termination === undefined) {
    return null;
  }
  return (
    <div className="runs-block runs-block-wide">
      <h4 className="runs-block-title">Terminal record</h4>
      {attempt.result === undefined ? null : (
        <div className="runs-terminal-record">
          <p>
            <strong>Planner result: {attempt.result.outcome}</strong>
          </p>
          <ResultSummary summary={attempt.result.summary} />
          {attempt.result.error === undefined ? null : (
            <p>
              Error <code>{attempt.result.error.code}</code> ·{" "}
              {retryability(attempt.result.error.retryable)}
              <br />
              {attempt.result.error.message}
            </p>
          )}
          {Object.keys(attempt.result.artifacts).length === 0 ? null : (
            <div className="runs-ref-list">
              {Object.entries(attempt.result.artifacts).map(
                ([slot, artifact]) => (
                  <RunArtifactRef
                    key={slot}
                    runId={runId}
                    slot={slot}
                    artifact={artifact}
                  />
                ),
              )}
            </div>
          )}
        </div>
      )}
      {attempt.termination === undefined ? null : (
        <div className="runs-terminal-record">
          <p>
            <strong>{attempt.termination.outcome}</strong> during{" "}
            {attempt.termination.phase} ·{" "}
            <code>{attempt.termination.code}</code>
          </p>
          <p>{attempt.termination.message}</p>
          <small>
            {retryability(attempt.termination.retryable)} ·{" "}
            {formatTimestamp(attempt.termination.occurredAt)}
          </small>
        </div>
      )}
    </div>
  );
}

function TransitionView({ transition }: { transition: StageTransition }) {
  return (
    <li className="runs-transition">
      <strong>{transition.action}</strong>
      {transition.targetStage === undefined ? null : (
        <span>
          Stage <code>{transition.targetStage}</code>
        </span>
      )}
      {transition.targetExecutionId === undefined ? null : (
        <span>
          execution{" "}
          <ExecutionId
            value={transition.targetExecutionId}
            label="target stage execution ID"
          />
        </span>
      )}
      {transition.escalationOrdinal === undefined ? null : (
        <span>escalation {transition.escalationOrdinal}</span>
      )}
      {transition.escalationExhausted ? (
        <span>escalation exhausted</span>
      ) : null}
      <small>{formatTimestamp(transition.decidedAt)}</small>
    </li>
  );
}

function AttemptFacts({ attempt }: { attempt: StageAttempt }) {
  const time = (value: string | undefined) =>
    value === undefined ? undefined : formatTimestamp(value);
  return (
    <Facts
      className="runs-inline-facts"
      items={[
        {
          term: "Stage execution ID",
          value: (
            <ExecutionId
              value={attempt.stageExecutionId}
              label="stage execution ID"
            />
          ),
        },
        { term: "Created", value: time(attempt.createdAt) },
        { term: "Updated", value: time(attempt.updatedAt) },
        { term: "Planner started", value: time(attempt.plannerStartedAt) },
        { term: "Terminal", value: time(attempt.terminalAt) },
      ]}
    />
  );
}

/** One Stage attempt of the Run page's Technical details. */
export function StageAttemptView({
  runId,
  attempt,
  active,
  projection,
  transitions,
  disclosure,
  instructionDisclosure,
}: {
  runId: string;
  attempt: StageAttempt;
  active: boolean;
  projection: PlannerProjection;
  transitions: StageTransition[];
  disclosure: RunDisclosureProps;
  instructionDisclosure: (subtaskId: string) => RunDisclosureProps;
}) {
  const state = stageStateLabel(attempt.state);
  return (
    <details
      className="run-attempt runs-attempt"
      id={`attempt-${attempt.stageExecutionId}`}
      {...disclosure}
    >
      <summary className="runs-attempt-summary">
        <DisclosureChevron />
        <StatusGlyph tone={state.tone} />
        <span className="runs-attempt-name">
          <code>{attempt.stage}</code>
          <span className="ui-caption">attempt {attempt.attempt}</span>
          {active ? <strong className="runs-marker">active</strong> : null}
        </span>
        <span className="runs-attempt-aside">
          <StatusChip tone={state.tone} size="sm" glyph={false}>
            {state.label}
          </StatusChip>
          <code title={attempt.stageExecutionId}>
            {compactId(attempt.stageExecutionId)}
          </code>
        </span>
      </summary>
      <div className="runs-attempt-body">
        <div className="runs-block runs-block-wide runs-objective">
          <p className="runs-label">Stage objective</p>
          <h3 className="runs-objective-title">
            {attempt.objective ?? "Unavailable before Stage preparation"}
          </h3>
          {attempt.previousExecutionId === undefined ? null : (
            <p className="runs-block-note">
              Continues{" "}
              <ExecutionId
                value={attempt.previousExecutionId}
                label="previous stage execution ID"
              />
            </p>
          )}
          <AttemptFacts attempt={attempt} />
        </div>
        <PlannerPlanView
          projection={projection}
          instructionDisclosure={instructionDisclosure}
        />
        <ExecutionConfigView attempt={attempt} />
        {attempt.metrics === undefined ? null : (
          <MetricsView metrics={attempt.metrics} />
        )}
        <RuntimeConfigurationView attempt={attempt} />
        <ResourceView attempt={attempt} />
        {attempt.diagnostics === undefined ? null : (
          <DiagnosticsView diagnostics={attempt.diagnostics} />
        )}
        <ResultView runId={runId} attempt={attempt} />
        {transitions.length === 0 ? null : (
          <div className="runs-block runs-block-wide">
            <h4 className="runs-block-title">Scheduler decisions</h4>
            <ol className="runs-transition-list">
              {transitions.map((transition) => (
                <TransitionView
                  key={`${transition.sourceExecutionId}-${transition.decidedAt}-${transition.action}`}
                  transition={transition}
                />
              ))}
            </ol>
          </div>
        )}
      </div>
    </details>
  );
}
