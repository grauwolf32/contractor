import { lazy, type ReactNode, Suspense, type SyntheticEvent } from "react";
import { Link } from "react-router";

import type { components } from "../../api/generated/public";
import { AllocationResourceList } from "../operations/performance/resources";
import type {
  StageAttempt,
  StageTransition,
  WorkflowRunState,
} from "../../api/runs";
import { compactDigest, compactId, formatTimestamp } from "../../app/format";
import { IdChip, StatusChip, StatusGlyph } from "../../ui";
import type { PlannerProjection } from "./live";
import { artifactDetailPath } from "../artifacts/paths";
import {
  allocationStatusLabel,
  runStateLabel,
  stageStateLabel,
  subtaskStatusLabel,
} from "./run-state";

type ArtifactRef = components["schemas"]["ExactArtifactRef"];
type ConsumerConfig = components["schemas"]["ConsumerExecutionConfig"];
type Metrics = components["schemas"]["MetricsSummary"];
type Diagnostics = components["schemas"]["AttemptDiagnostics"];

const MarkdownPreview = lazy(() => import("../artifacts/previews/markdown"));

function stateLabel(value: string): string {
  return value.replaceAll("_", " ");
}

/** Controlled `<details>` state shared by the heavy Run detail sections. */
export type RunDisclosureProps = {
  open: boolean;
  onToggle: (event: SyntheticEvent<HTMLDetailsElement>) => void;
};

/** The rotating chevron of the Runs disclosures. */
export function DisclosureChevron() {
  return (
    <svg
      className="runs-section-chevron"
      width="13"
      height="13"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d="M9.5 6l6 6-6 6" />
    </svg>
  );
}

/**
 * One collapsible section of the Run page's Technical details: a native
 * `<details>` whose summary row carries the title (an h3, which the summary
 * content model allows), a quiet description and a count at the right.
 * `open` is controlled by the page so choices survive live remounts.
 */
export function RunSection({
  id,
  className,
  title,
  description,
  aside,
  children,
  ...disclosure
}: {
  id?: string | undefined;
  className?: string | undefined;
  title: string;
  description?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
} & RunDisclosureProps) {
  return (
    <details
      id={id}
      className={["runs-section", className].filter(Boolean).join(" ")}
      {...disclosure}
    >
      <summary className="runs-section-summary">
        <DisclosureChevron />
        <h3 className="runs-section-title">{title}</h3>
        {description === undefined ? null : (
          <span className="runs-section-description">{description}</span>
        )}
        {aside === undefined ? null : (
          <span className="runs-section-aside">{aside}</span>
        )}
      </summary>
      <div className="runs-section-body">{children}</div>
    </details>
  );
}

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
 * Lifecycle badge shared by several areas (Projects, Checks, Evals,
 * Operations). The Runs pages use RunStateChip instead.
 */
export function StateBadge({ state }: { state: WorkflowRunState | string }) {
  return (
    <span className={`state-badge state-${state.replaceAll("_", "-")}`}>
      {stateLabel(state)}
    </span>
  );
}

/** WorkflowRun state as a V3B status chip: glyph and word. */
export function RunStateChip({
  state,
  size = "sm",
}: {
  state: WorkflowRunState;
  size?: "sm" | "md";
}) {
  const { label, tone } = runStateLabel(state);
  return (
    <StatusChip tone={tone} size={size}>
      {label}
    </StatusChip>
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
      to={artifactDetailPath({ kind: "run", id: runId }, artifact)}
    >
      {slot === undefined ? null : (
        <>
          <strong>{slot}</strong>{" "}
        </>
      )}
      <code>
        {artifact.namespace}/{artifact.name}@{artifact.revision}
      </code>
    </Link>
  );
}

function Digest({ value }: { value: string }) {
  return (
    <code className="runs-digest" title={value}>
      {compactDigest(value)}
    </code>
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

function ConsumerConfigView({
  label,
  config,
}: {
  label: string;
  config: ConsumerConfig;
}) {
  if (config.modelPolicy === undefined) {
    return (
      <li>
        <strong>{label}</strong>
        <span>Tool execution · no model</span>
      </li>
    );
  }
  return (
    <li>
      <strong>{label}</strong>
      <span>
        ModelPolicy{" "}
        <code>
          {config.modelPolicy.policyId}@{config.modelPolicy.version}
        </code>{" "}
        <Digest value={config.modelPolicy.digest} />
      </span>
      {config.llmGateway === undefined ? (
        <span>LLMGatewayConfig resolved during Runtime placement</span>
      ) : (
        <span>
          LLMGatewayConfig{" "}
          <code>
            {config.llmGateway.gatewayId}@{config.llmGateway.version}
          </code>{" "}
          <Digest value={config.llmGateway.digest} />
        </span>
      )}
      <span>Credential {config.credential?.credentialId ?? "none"}</span>
      <span className="runs-config-origins">
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
      <ul className="runs-consumer-list">
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
  if (attempt.runtimeConfigurationUnavailable) {
    return (
      <p className="notice notice-warning" role="status">
        Runtime configuration details could not be verified. Results remain
        available. Refresh this Run to retry.
      </p>
    );
  }
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
          return (
            <article className="runs-allocation" key={allocation.logicalAgent}>
              <div className="runs-allocation-head">
                <strong>Logical Worker {allocation.logicalAgent}</strong>
                <StatusChip tone={status.tone} size="sm">
                  {status.label}
                </StatusChip>
              </div>
              <dl className="runs-facts-list">
                <div>
                  <dt>Final Agent labels</dt>
                  <dd>
                    {allocation.agentLabels.length === 0
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
                      <ul className="runs-origin-list">
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
    <div className="runs-block">
      <h4 className="runs-block-title">Safe aggregate metrics</h4>
      <dl className="runs-metrics">
        {values.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
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
              {attempt.result.error.retryable ? "retryable" : "not retryable"}
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
  const values: Array<[string, string | undefined]> = [
    ["Created", attempt.createdAt],
    ["Updated", attempt.updatedAt],
    ["Planner started", attempt.plannerStartedAt],
    ["Terminal", attempt.terminalAt],
  ];
  return (
    <dl className="runs-inline-facts">
      <div>
        <dt>Stage execution ID</dt>
        <dd>
          <ExecutionId
            value={attempt.stageExecutionId}
            label="stage execution ID"
          />
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
          <span className="runs-attempt-number">attempt {attempt.attempt}</span>
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

export function DefinitionList({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="runs-binding">
      <p className="runs-label">{title}</p>
      {children}
    </div>
  );
}
