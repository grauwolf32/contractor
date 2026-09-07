import type { ReactNode } from "react";
import { Link } from "react-router";

import type { components } from "../../api/generated/public";
import { AllocationResourceList } from "../operations/performance/resources";
import type {
  StageAttempt,
  StageTransition,
  WorkflowRunState,
} from "../../api/runs";
import { formatTimestamp } from "../artifacts/common";
import type { PlannerProjection } from "./live";

type ArtifactRef = components["schemas"]["ExactArtifactRef"];
type ConsumerConfig = components["schemas"]["ConsumerExecutionConfig"];
type Metrics = components["schemas"]["MetricsSummary"];
type Diagnostics = components["schemas"]["AttemptDiagnostics"];

function stateLabel(value: string): string {
  return value.replaceAll("_", " ");
}

export function StateBadge({ state }: { state: WorkflowRunState | string }) {
  return (
    <span className={`state-badge state-${state.replaceAll("_", "-")}`}>
      {stateLabel(state)}
    </span>
  );
}

export function RunMetadataLabelChips({
  labels,
  empty = "none",
}: {
  labels: Readonly<Record<string, string>>;
  empty?: string;
}) {
  const entries = Object.entries(labels).sort(([left], [right]) =>
    left === right ? 0 : left < right ? -1 : 1,
  );
  if (entries.length === 0) {
    return <span className="muted-copy">{empty}</span>;
  }
  return (
    <span className="run-metadata-label-chips">
      {entries.map(([key, value]) => (
        <code className="run-metadata-label-chip" key={key} title={value}>
          <strong>{key}</strong>
          {value === "" ? null : (
            <>
              :<span>{value}</span>
            </>
          )}
        </code>
      ))}
    </span>
  );
}

function compactDigest(digest: string): string {
  return `${digest.slice(0, 14)}…${digest.slice(-8)}`;
}

export function RunArtifactRef({
  runId,
  slot,
  artifact,
}: {
  runId: string;
  slot?: string;
  artifact: ArtifactRef;
}) {
  return (
    <Link
      className="artifact-ref-link"
      to={`/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifact.namespace)}/${encodeURIComponent(artifact.name)}?revision=${encodeURIComponent(artifact.revision)}`}
    >
      {slot === undefined ? null : <strong>{slot}</strong>}
      <code>
        {artifact.namespace}/{artifact.name}@{artifact.revision}
      </code>
    </Link>
  );
}

function ConsumerConfigView({
  label,
  config,
}: {
  label: string;
  config: ConsumerConfig;
}) {
  return (
    <li>
      <strong>{label}</strong>
      <span>
        ModelPolicy {config.modelPolicy.policyId}@{config.modelPolicy.version} ·{" "}
        <small title={config.modelPolicy.digest}>
          {compactDigest(config.modelPolicy.digest)}
        </small>
      </span>
      {config.llmGateway === undefined ? (
        <span>LLMGatewayConfig resolved during Runtime placement</span>
      ) : (
        <span>
          LLMGatewayConfig {config.llmGateway.gatewayId}@
          {config.llmGateway.version} ·{" "}
          <small title={config.llmGateway.digest}>
            {compactDigest(config.llmGateway.digest)}
          </small>
        </span>
      )}
      <span>Credential {config.credential?.credentialId ?? "none"}</span>
      <span className="config-origins">
        Origins: model {config.origins?.modelPolicy ?? "not reported"}; Gateway{" "}
        {config.origins?.llmGateway ?? "not reported"}; credential{" "}
        {config.origins?.credential ?? "not reported"}
      </span>
    </li>
  );
}

function ExecutionConfigView({ attempt }: { attempt: StageAttempt }) {
  const config = attempt.executionConfig;
  return (
    <div className="attempt-block">
      <h4>Resolved execution configuration</h4>
      <p className="compact-copy">
        Variant <code>{config.variant}</code>
        {config.escalationOrdinal === undefined
          ? null
          : ` · escalation ${config.escalationOrdinal}`}
      </p>
      {config.ref === undefined ? null : (
        <p className="compact-copy">
          Profile{" "}
          <code>
            {config.ref.configId}@{config.ref.version}
          </code>{" "}
          ·{" "}
          <small title={config.ref.digest}>
            {compactDigest(config.ref.digest)}
          </small>
        </p>
      )}
      <ul className="consumer-config-list">
        {config.planner === undefined ? null : (
          <ConsumerConfigView label="Planner" config={config.planner} />
        )}
        {Object.entries(config.agents)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([name, consumer]) => (
            <ConsumerConfigView
              key={name}
              label={`Logical Worker ${name}`}
              config={consumer}
            />
          ))}
      </ul>
    </div>
  );
}

function RuntimeConfigurationView({ attempt }: { attempt: StageAttempt }) {
  const configuration = attempt.runtimeConfiguration;
  if (configuration === undefined) {
    return null;
  }
  return (
    <div className="attempt-block stage-runtime-configuration">
      <h4>Allocation-pinned Runtime configuration</h4>
      <p className="compact-copy">
        This historical projection appears only after placement commits. It
        contains safe refs and origins, never RuntimeSettings or physical Agent
        identity.
      </p>
      <div className="stage-runtime-allocation-list">
        {configuration.allocations.map((allocation) => (
          <article key={allocation.logicalAgent}>
            <div className="section-heading">
              <strong>Logical Worker {allocation.logicalAgent}</strong>
              <StateBadge state={allocation.status} />
            </div>
            <dl className="key-value-list">
              <div>
                <dt>Final Agent labels</dt>
                <dd>
                  {allocation.agentLabels.length === 0
                    ? "none"
                    : allocation.agentLabels.map((pin) => (
                        <span className="runtime-agent-pin" key={pin.label}>
                          <strong>{pin.label}</strong>
                          <code>
                            {pin.config.name}@{pin.config.version}
                          </code>
                          <small>binding revision {pin.bindingRevision}</small>
                          <code title={pin.config.digest}>
                            {compactDigest(pin.config.digest)}
                          </code>
                        </span>
                      ))}
                </dd>
              </div>
              <div>
                <dt>Required adapters</dt>
                <dd>
                  {allocation.runtimeAdapters.length === 0
                    ? "none"
                    : allocation.runtimeAdapters.map((adapter) => (
                        <code key={adapter}>{adapter}</code>
                      ))}
                </dd>
              </div>
              <div>
                <dt>Final safe origins</dt>
                <dd>
                  {Object.entries(allocation.origins).length === 0 ? (
                    "none"
                  ) : (
                    <ul className="runtime-origin-list">
                      {Object.entries(allocation.origins).map(
                        ([field, origin]) => (
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
                        ),
                      )}
                    </ul>
                  )}
                </dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
    </div>
  );
}

function PlannerPlanView({ projection }: { projection: PlannerProjection }) {
  const plan = projection.plan;
  if (plan === undefined) {
    return (
      <div className="compact-empty">
        No durable subtask plan is present for this attempt.
      </div>
    );
  }
  const dispatch = projection.dispatch ?? plan.activeDispatch;
  const dispatchPhase = projection.dispatch?.phase ?? "running";
  return (
    <div className="planner-projection">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Nested Planner work</p>
          <h4>Subtasks</h4>
        </div>
        <span>revision {plan.revision}</span>
      </div>
      {dispatch === undefined ? null : (
        <div className="dispatch-banner">
          <strong>Logical Worker {dispatch.workerName}</strong>
          <span>
            subtask {dispatch.subtaskId} · {dispatchPhase}
          </span>
          <code>{dispatch.callId}</code>
        </div>
      )}
      <ol className="subtask-list">
        {plan.subtasks.map((subtask) => {
          const current = plan.currentSubtaskId === subtask.id;
          return (
            <li className={current ? "current" : undefined} key={subtask.id}>
              <div>
                <code>{subtask.id}</code>
                <StateBadge state={subtask.status} />
                {current ? (
                  <strong className="current-marker">current</strong>
                ) : null}
              </div>
              <p>{subtask.objective}</p>
              <details>
                <summary>Planner instructions</summary>
                <p>{subtask.instructions}</p>
              </details>
            </li>
          );
        })}
      </ol>
      {projection.lastEventKind === undefined ? null : (
        <small className="live-event-note">
          Last typed fact: {projection.lastEventKind}
          {projection.lastOccurredAt === undefined
            ? ""
            : ` · ${formatTimestamp(projection.lastOccurredAt)}`}
        </small>
      )}
    </div>
  );
}

function MetricsView({ metrics }: { metrics: Metrics }) {
  const values = [
    ["Model calls", metrics.modelCalls],
    ["Input tokens", metrics.inputTokens],
    ["Output tokens", metrics.outputTokens],
    ["Total tokens", metrics.totalTokens],
    ["Tool calls", metrics.toolCalls],
    ["Tool failures", metrics.toolFailures],
    ["Errors", metrics.errorCount],
  ] as const;
  return (
    <div className="attempt-block">
      <h4>Safe aggregate metrics</h4>
      <dl className="metrics-grid">
        {values.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <p className="compact-copy">
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
    <div className="attempt-block run-attempt-resources">
      <h4>Allocation resource observations</h4>
      <p className="compact-copy">
        These terminal summaries use the allocation-pinned collection policy;
        missing measurements are not zero usage.
      </p>
      <AllocationResourceList items={attempt.resources} showRunLink={false} />
    </div>
  );
}

function DiagnosticsView({ diagnostics }: { diagnostics: Diagnostics }) {
  return (
    <div className="attempt-block">
      <h4>Attempt diagnostics</h4>
      {diagnostics.items.length === 0 ? (
        <p className="compact-copy">
          No normalized Planner or Worker errors were reported.
        </p>
      ) : (
        <ol className="diagnostic-list">
          {diagnostics.items.map((diagnostic, index) => (
            <li
              key={`${diagnostic.participant}-${diagnostic.logicalAgent ?? "planner"}-${diagnostic.code}-${index}`}
            >
              <div>
                <strong>
                  {diagnostic.participant === "planner" ? "Planner" : "Worker"}
                </strong>
                {diagnostic.logicalAgent === undefined ? null : (
                  <code>{diagnostic.logicalAgent}</code>
                )}
                <code>{diagnostic.code}</code>
                <span>
                  {diagnostic.retryable === undefined
                    ? "retryability unknown"
                    : diagnostic.retryable
                      ? "retryable"
                      : "not retryable"}
                </span>
              </div>
              <p>{diagnostic.message}</p>
            </li>
          ))}
        </ol>
      )}
      {diagnostics.truncated ? (
        <p className="compact-copy">
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
    <div className="attempt-block">
      <h4>Terminal record</h4>
      {attempt.result === undefined ? null : (
        <div className="result-record">
          <p>
            <strong>Planner result: {attempt.result.outcome}</strong>
          </p>
          <p>{attempt.result.summary}</p>
          {attempt.result.error === undefined ? null : (
            <p>
              Error <code>{attempt.result.error.code}</code> ·{" "}
              {attempt.result.error.retryable ? "retryable" : "not retryable"}
              <br />
              {attempt.result.error.message}
            </p>
          )}
          {Object.entries(attempt.result.artifacts).map(([slot, artifact]) => (
            <RunArtifactRef
              key={slot}
              runId={runId}
              slot={slot}
              artifact={artifact}
            />
          ))}
        </div>
      )}
      {attempt.termination === undefined ? null : (
        <div className="termination-record">
          <p>
            <strong>{attempt.termination.outcome}</strong> during{" "}
            {attempt.termination.phase} ·{" "}
            <code>{attempt.termination.code}</code>
          </p>
          <p>{attempt.termination.message}</p>
          <small>
            {attempt.termination.retryable ? "retryable" : "not retryable"} ·{" "}
            {formatTimestamp(attempt.termination.occurredAt)}
          </small>
        </div>
      )}
    </div>
  );
}

function TransitionView({ transition }: { transition: StageTransition }) {
  return (
    <li>
      <strong>{transition.action}</strong>
      {transition.targetStage === undefined ? null : (
        <span>
          Stage <code>{transition.targetStage}</code>
        </span>
      )}
      {transition.targetExecutionId === undefined ? null : (
        <span>
          execution <code>{transition.targetExecutionId}</code>
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

function AttemptTimestamps({ attempt }: { attempt: StageAttempt }) {
  const values: Array<[string, string | undefined]> = [
    ["created", attempt.createdAt],
    ["updated", attempt.updatedAt],
    ["Planner started", attempt.plannerStartedAt],
    ["terminal", attempt.terminalAt],
  ];
  return (
    <dl className="attempt-timestamps">
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

export function StageAttemptView({
  runId,
  attempt,
  active,
  focused,
  projection,
  transitions,
}: {
  runId: string;
  attempt: StageAttempt;
  active: boolean;
  focused: boolean;
  projection: PlannerProjection;
  transitions: StageTransition[];
}) {
  return (
    <details
      className="run-attempt"
      id={`attempt-${attempt.stageExecutionId}`}
      open={active || focused}
    >
      <summary>
        <span>
          <code>{attempt.stage}</code>
          <strong>attempt {attempt.attempt}</strong>
          {active ? <span className="current-marker">active</span> : null}
        </span>
        <span>
          <StateBadge state={attempt.state} />
          <code>{attempt.stageExecutionId}</code>
        </span>
      </summary>
      <div className="run-attempt-body">
        <div className="attempt-block global-task">
          <p className="eyebrow">Immutable Stage objective · global task</p>
          <h4>{attempt.objective ?? "Unavailable before Stage preparation"}</h4>
          {attempt.previousExecutionId === undefined ? null : (
            <p className="compact-copy">
              Continues <code>{attempt.previousExecutionId}</code>
            </p>
          )}
          <AttemptTimestamps attempt={attempt} />
        </div>
        <PlannerPlanView projection={projection} />
        <ExecutionConfigView attempt={attempt} />
        <RuntimeConfigurationView attempt={attempt} />
        {attempt.metrics === undefined ? null : (
          <MetricsView metrics={attempt.metrics} />
        )}
        <ResourceView attempt={attempt} />
        {attempt.diagnostics === undefined ? null : (
          <DiagnosticsView diagnostics={attempt.diagnostics} />
        )}
        <ResultView runId={runId} attempt={attempt} />
        {transitions.length === 0 ? null : (
          <div className="attempt-block">
            <h4>Scheduler decisions</h4>
            <ol className="transition-list">
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

export function DefinitionList({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="panel run-definition-panel">
      <p className="eyebrow">{title}</p>
      {children}
    </div>
  );
}
