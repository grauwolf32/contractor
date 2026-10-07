import "./runs.css";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ReactNode,
  type Ref,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import { Link, useLocation, useParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  getRun,
  isTerminalRunState,
  RUN_ID_PATTERN,
  type RunStatus,
} from "../../api/runs";
import { useSession } from "../../auth/session";
import {
  catalogReturnState,
  runtimeConfigVersionPath,
} from "../../app/navigation";
import { ErrorNotice } from "../../app/error-notice";
import { compactDigest, compactId, formatTimestamp } from "../../app/format";
import { StaleDataWarning } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import { useDocumentTitle } from "../../app/document-title";
import { RecordedTime } from "../../app/recorded-time";
import { IdChip, StatusChip, StatusGlyph, type StatusTone } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { RunActions } from "./actions";
import { RunArtifactLibrary, RunOutputGallery } from "./artifacts";
import {
  DefinitionList,
  RunArtifactRef,
  type RunDisclosureProps,
  RunMetadataLabelChips,
  RunSection,
  RunStateChip,
  StageAttemptView,
} from "./components";
import { type LiveRunProjection, useLiveRunProjection } from "./live";
import { RecoveryStatus } from "./recovery";
import { parseWorkflowIdentity } from "./output-model";
import { publicationStatusLabel, runStateLabel } from "./run-state";
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

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
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
    case "pending":
      return "Run is queued";
    case "waiting":
      return "Waiting for the model";
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
    case "pending":
      return "Execution will start when resources and the model are available.";
    case "waiting":
      return "The current invocation is preserved while the model recovers.";
    case "initializing":
      return "Preparing to start the first stage.";
    case "running":
      return "Follow the current stage and its progress under Technical details.";
  }
}

function retryabilityLabel(issue: NonNullable<RunTriage["issue"]>): {
  label: string;
  tone: StatusTone;
} {
  if (issue.source === "cancellation") {
    return { label: "User requested", tone: "neutral" };
  }
  if (issue.retryable === undefined) {
    return { label: "Retryability unknown", tone: "neutral" };
  }
  return issue.retryable
    ? { label: "Retryable", tone: "warning" }
    : { label: "Not retryable", tone: "blocked" };
}

function RunTriageSummary({
  run,
  triage,
  onOpenAttempt,
}: {
  run: RunStatus;
  triage: RunTriage;
  onOpenAttempt: (stageExecutionId: string) => void;
}) {
  const { session } = useSession();
  const canOperate =
    session?.principal.capabilities.includes("operations") === true;
  const focusedAttempt = triage.stageExecutionId;
  const state = runStateLabel(run.state);
  const issue = triage.issue;
  const retry = issue === undefined ? undefined : retryabilityLabel(issue);
  return (
    <section
      className={`runs-triage run-triage run-triage-${run.state}`}
      aria-labelledby="run-triage-title"
    >
      <div className="runs-triage-head">
        <StatusGlyph tone={state.tone} size={20} />
        <div className="runs-triage-heading">
          <h2 className="runs-h2" id="run-triage-title">
            {triageTitle(run, triage)}
          </h2>
          {issue === undefined ? (
            <p className="runs-triage-text">{triageDescription(run, triage)}</p>
          ) : null}
        </div>
      </div>
      {issue === undefined || retry === undefined ? null : (
        <div className="runs-cause" data-source={issue.source}>
          <p className="runs-label">
            {issue.source === "cancellation"
              ? "Lifecycle reason"
              : "Primary cause"}
          </p>
          <div className="runs-cause-line">
            <code>{issue.code}</code>
            <StatusChip tone={retry.tone} size="sm" glyph={false}>
              {retry.label}
            </StatusChip>
          </div>
          <p className="runs-cause-message">{triageDescription(run, triage)}</p>
          {issue.participant === undefined ? null : (
            <small>
              Reported by {issue.participant}
              {issue.logicalAgent === undefined ? "" : ` ${issue.logicalAgent}`}
            </small>
          )}
        </div>
      )}
      <RecoveryStatus run={run} />
      <div className="runs-next" data-kind={triage.guidance.kind}>
        <p className="runs-label">Next step</p>
        <p className="runs-next-title">{triage.guidance.title}</p>
        <p className="runs-next-text">{triage.guidance.message}</p>
        {run.state !== "failed" ? null : run.resumeStageExecutionId ===
          undefined ? (
          <p className="runs-next-text">
            Continuation is unavailable: cleanup must finish and the Run must
            have a failed stage. Audit-managed and evaluation Runs use their own
            lifecycle controls.
          </p>
        ) : (
          <p className="runs-next-text">
            Continue from failed stage keeps the successful stages and their
            results, and gives the failed stage a new attempt.
          </p>
        )}
        {(run.state === "succeeded" && triage.outputCount > 0) ||
        (triage.guidance.operationsPath !== undefined && canOperate) ? (
          <div className="runs-next-links">
            {run.state === "succeeded" && triage.outputCount > 0 ? (
              <a
                className="ui-btn"
                data-size="sm"
                data-variant="primary"
                href="#run-outputs"
              >
                Read results
              </a>
            ) : null}
            {triage.guidance.operationsPath === undefined ||
            !canOperate ? null : (
              <Link
                className="ui-btn"
                data-size="sm"
                to={triage.guidance.operationsPath}
              >
                Open Operations diagnostics
              </Link>
            )}
          </div>
        ) : null}
      </div>
      {focusedAttempt === undefined && triage.outputCount === 0 ? null : (
        <nav className="runs-shortcuts" aria-label="Run triage shortcuts">
          {focusedAttempt === undefined ? null : (
            <a
              href={`#attempt-${focusedAttempt}`}
              onClick={() => onOpenAttempt(focusedAttempt)}
            >
              Inspect focused attempt
            </a>
          )}
          {triage.outputCount === 0 ? null : (
            <a href="#run-outputs">Preview outputs</a>
          )}
        </nav>
      )}
    </section>
  );
}

function RunOutputPublications({ run }: { run: RunStatus }) {
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
        <p className="runs-hint">
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

function RunExecutionMetrics({
  run,
  triage,
  ...disclosure
}: { run: RunStatus; triage: RunTriage } & RunDisclosureProps) {
  return (
    <RunSection
      id="run-metrics"
      title="Execution metrics"
      description="Duration, attempts, tokens and tool calls"
      aside={`${formatRunDuration(triage.durationMs)} · ${plural(triage.attemptCount, "attempt")}`}
      {...disclosure}
    >
      <dl className="runs-metric-tiles">
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

function RunBindings({
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
            <p className="runs-hint">None supplied.</p>
          ) : (
            <dl className="runs-facts-list">
              {parameters.map(([name, value]) => (
                <div key={name}>
                  <dt>{name}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          )}
        </DefinitionList>
        <DefinitionList title="Input revisions">
          {inputs.length === 0 ? (
            <p className="runs-hint">No inputs.</p>
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

function RunMetadataLabels({
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
      <p className="runs-hint">
        Use these labels to find related Runs. Labels are fixed at creation.
      </p>
      <RunMetadataLabelChips labels={run.labels} empty="No metadata labels." />
    </RunSection>
  );
}

function RunRuntimeConfiguration({
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
      <p className="runs-hint">
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
            <code className="runs-digest" title={pin.config.digest}>
              {compactDigest(pin.config.digest)}
            </code>
          </article>
        ))}
      </div>
      <p className="runs-hint">
        Agent-label overrides become knowable only after Scheduler commits an
        allocation snapshot; they are never inferred from current Operations
        state.
      </p>
    </RunSection>
  );
}

function RunCancellationRecord({
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
      <dl className="runs-facts-list">
        <div>
          <dt>Code</dt>
          <dd>
            <code>{cancellation.code}</code>
          </dd>
        </div>
        <div>
          <dt>Requested</dt>
          <dd>{formatTimestamp(cancellation.requestedAt)}</dd>
        </div>
        {cancellation.requestedBy === undefined ? null : (
          <div>
            <dt>Requested by</dt>
            <dd>{cancellation.requestedBy}</dd>
          </div>
        )}
        {cancellation.reason === undefined ? null : (
          <div>
            <dt>Reason</dt>
            <dd>{cancellation.reason}</dd>
          </div>
        )}
      </dl>
    </RunSection>
  );
}

type LiveStatus = Pick<
  LiveRunProjection,
  "connection" | "error" | "resyncReason"
>;

function sameLiveStatus(left: LiveStatus, right: LiveStatus): boolean {
  return (
    left.connection === right.connection &&
    left.error === right.error &&
    left.resyncReason === right.resyncReason
  );
}

function LiveAttempts({
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
        <p className="runs-hint">
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

const LIVE_TONES: Record<LiveStatus["connection"], StatusTone> = {
  live: "success",
  connecting: "progress",
  reconnecting: "warning",
  resyncing: "warning",
  error: "blocked",
  unavailable: "neutral",
};

function LiveStatusLine({
  status,
  terminal,
}: {
  status: LiveStatus;
  terminal: boolean;
}) {
  // A finished Run without an event stream has nothing to follow.
  if (terminal && status.connection === "unavailable") return null;
  return (
    <p className="runs-live" data-connection={status.connection} role="status">
      <StatusGlyph tone={LIVE_TONES[status.connection]} size={14} />
      <span>
        Live events: {status.connection}
        {status.resyncReason === undefined
          ? null
          : ` · REST resync after ${status.resyncReason.replaceAll("_", " ")}`}
      </span>
    </p>
  );
}

function BreadcrumbSeparator() {
  return (
    <svg
      className="runs-breadcrumb-separator"
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
 * "Completed › run_0123…": the Runs view the Run belongs to, or the page the
 * user opened it from (with that page's own return state). The rail already
 * marks Runs as the destination, so the trail does not repeat it.
 */
function RunBreadcrumb({
  runId,
  terminal,
}: {
  runId: string;
  terminal: boolean;
}) {
  const { state } = useLocation();
  const origin = catalogReturnState(
    state,
    terminal
      ? { returnTo: "/runs?view=completed", returnLabel: "Completed" }
      : { returnTo: "/runs", returnLabel: "Queue" },
  );
  return (
    <nav aria-label="Breadcrumb" className="runs-breadcrumb">
      <ol>
        <li>
          <Link to={origin.returnTo} state={origin.returnState}>
            {origin.returnLabel}
          </Link>
          <BreadcrumbSeparator />
        </li>
        <li>
          <span aria-current="page" title={runId}>
            {compactId(runId)}
          </span>
        </li>
      </ol>
    </nav>
  );
}

function RunTitle({
  workflow,
  headingRef,
}: {
  workflow: string;
  headingRef?: Ref<HTMLHeadingElement>;
}) {
  const identity = parseWorkflowIdentity(workflow);
  return (
    <h1
      className="runs-run-title"
      id="run-title"
      ref={headingRef}
      tabIndex={-1}
    >
      {identity === undefined ? (
        workflow
      ) : (
        <>
          {identity.name}
          <span className="runs-run-version">@{identity.version}</span>
        </>
      )}
    </h1>
  );
}

function RunFacts({
  run,
  durationMs,
}: {
  run: RunStatus;
  durationMs?: number;
}) {
  return (
    <dl className="runs-facts run-metadata">
      <div>
        <dt>Run ID</dt>
        <dd>
          <IdChip value={run.runId} label="Run ID" />
        </dd>
      </div>
      <div>
        <dt>Workflow</dt>
        <dd>
          <IdChip
            value={run.workflow}
            display={run.workflow}
            label="workflow version"
          />
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
      {run.createdAt === undefined ? null : (
        <div>
          <dt>Created</dt>
          <dd>{formatTimestamp(run.createdAt)}</dd>
        </div>
      )}
      {run.startedAt === undefined ? null : (
        <div>
          <dt>Started</dt>
          <dd>{formatTimestamp(run.startedAt)}</dd>
        </div>
      )}
      {run.updatedAt === undefined ? null : (
        <div>
          <dt>Updated</dt>
          <dd>{formatTimestamp(run.updatedAt)}</dd>
        </div>
      )}
      {run.finishedAt === undefined ? null : (
        <div>
          <dt>Finished</dt>
          <dd>{formatTimestamp(run.finishedAt)}</dd>
        </div>
      )}
      {durationMs === undefined ? null : (
        <div>
          <dt>Duration</dt>
          <dd>{formatRunDuration(durationMs)}</dd>
        </div>
      )}
    </dl>
  );
}

function LoadedRunDetail({
  run,
  snapshotVersion,
  refresh,
  staleError,
  onRetry,
  retryPending,
}: {
  run: RunStatus;
  snapshotVersion: number;
  refresh: ReactNode;
  staleError: Error | null;
  onRetry: () => void;
  retryPending: boolean;
}) {
  const queryClient = useQueryClient();
  const heading = useRef<HTMLHeadingElement>(null);
  const [announcement, setAnnouncement] = useState("");
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
  const terminal = isTerminalRunState(run.state);
  const [live, setLive] = useState<LiveStatus>(() => ({
    connection: run.eventCursor === undefined ? "unavailable" : "connecting",
  }));
  const reportLive = useCallback((next: LiveStatus) => {
    setLive((current) => (sameLiveStatus(current, next) ? current : next));
  }, []);
  // Heavy sections stay open while a Run is active or failed (diagnostics
  // matter) and start collapsed once it ended otherwise; reference material
  // always starts collapsed. Choices persist across live remounts because
  // this component is keyed by Run ID.
  const [disclosures, setDisclosures] = useState(() => {
    const open = !terminal || run.state === "failed";
    return {
      metrics: run.state !== "succeeded",
      attempts: open,
      labels: false,
      bindings: false,
      runtime: false,
      artifacts: open,
      cancellation: false,
    };
  });
  const [attemptOverrides, setAttemptOverrides] = useState<
    Record<string, boolean>
  >({});
  const [instructionOverrides, setInstructionOverrides] = useState<
    Record<string, boolean>
  >({});
  function disclosure(key: keyof typeof disclosures): RunDisclosureProps {
    return {
      open: disclosures[key],
      onToggle: (event) => {
        const next = event.currentTarget.open;
        setDisclosures((current) =>
          current[key] === next ? current : { ...current, [key]: next },
        );
      },
    };
  }
  function attemptDisclosure(
    stageExecutionId: string,
    defaultOpen: boolean,
  ): RunDisclosureProps {
    const open = attemptOverrides[stageExecutionId] ?? defaultOpen;
    return {
      open,
      onToggle: (event) => {
        const next = event.currentTarget.open;
        if (next !== open) {
          setAttemptOverrides((current) => ({
            ...current,
            [stageExecutionId]: next,
          }));
        }
      },
    };
  }
  function instructionDisclosure(
    stageExecutionId: string,
    subtaskId: string,
  ): RunDisclosureProps {
    const key = JSON.stringify([stageExecutionId, subtaskId]);
    const open = instructionOverrides[key] ?? false;
    return {
      open,
      onToggle: (event) => {
        const next = event.currentTarget.open;
        if (next !== open) {
          setInstructionOverrides((current) => ({ ...current, [key]: next }));
        }
      },
    };
  }
  // The triage shortcut opens the attempt (and its section) before the
  // browser follows the link to it.
  function openAttempt(stageExecutionId: string): void {
    flushSync(() => {
      setDisclosures((current) =>
        current.attempts ? current : { ...current, attempts: true },
      );
      setAttemptOverrides((current) =>
        current[stageExecutionId] === true
          ? current
          : { ...current, [stageExecutionId]: true },
      );
    });
  }
  function focusHeading(): void {
    // The Dialog returns focus to its trigger first; the trigger is usually
    // gone with the new state, so the page heading takes focus instead.
    window.setTimeout(() => heading.current?.focus(), 0);
  }
  function actionDone(message: string): void {
    setAnnouncement(message);
    focusHeading();
  }

  return (
    <>
      <header className="runs-run-header">
        <RunBreadcrumb runId={run.runId} terminal={terminal} />
        <div className="runs-run-titles">
          <p className="runs-label runs-run-kind">Workflow Run</p>
          <RunTitle workflow={run.workflow} headingRef={heading} />
          <RunStateChip state={run.state} size="md" />
        </div>
        <RunActions
          run={run}
          refresh={refresh}
          onDone={actionDone}
          focusHeading={focusHeading}
        />
        <RunFacts
          run={run}
          {...(triage.durationMs === undefined
            ? {}
            : { durationMs: triage.durationMs })}
        />
        <LiveStatusLine status={live} terminal={terminal} />
        <p className="ui-visually-hidden" role="status">
          {announcement}
        </p>
      </header>
      <div className="runs-run-body">
        {staleError === null ? null : (
          <StaleDataWarning
            error={staleError}
            onRetry={onRetry}
            retryPending={retryPending}
          />
        )}
        {live.error === undefined ? null : (
          <div className="notice notice-warning" role="alert">
            <strong>{live.error}</strong>
            <p>Use Refresh to reload the Run.</p>
          </div>
        )}
        <RunTriageSummary
          run={run}
          triage={triage}
          onOpenAttempt={openAttempt}
        />
        <RunOutputGallery run={run} />
        <RunOutputPublications run={run} />
        <section
          className="runs-technical runs-block-section"
          aria-labelledby="run-technical-title"
        >
          <div className="runs-section-head">
            <div>
              <h2 className="runs-h2" id="run-technical-title">
                Technical details
              </h2>
              <p className="runs-hint">
                Stages, attempts, configuration and files, for admins and
                debugging.
              </p>
            </div>
          </div>
          <div className="runs-sections">
            <RunExecutionMetrics
              run={run}
              triage={triage}
              {...disclosure("metrics")}
            />
            <RunSection
              id="run-attempts"
              title="Ordered Stage attempts"
              description="Scheduler history: Planner subtasks, configuration, diagnostics and records"
              aside={plural(run.attempts.length, "attempt")}
              {...disclosure("attempts")}
            >
              <LiveAttempts
                key={`${liveKey}:${snapshotVersion}`}
                run={run}
                focusStageExecutionId={triage.stageExecutionId}
                attemptDisclosure={attemptDisclosure}
                instructionDisclosure={instructionDisclosure}
                onLiveStatus={reportLive}
              />
            </RunSection>
            <RunMetadataLabels run={run} {...disclosure("labels")} />
            <RunBindings run={run} {...disclosure("bindings")} />
            <RunRuntimeConfiguration run={run} {...disclosure("runtime")} />
            <RunArtifactLibrary
              runId={run.runId}
              {...disclosure("artifacts")}
            />
            {run.cancellation === undefined ? null : (
              <RunCancellationRecord
                cancellation={run.cancellation}
                {...disclosure("cancellation")}
              />
            )}
          </div>
        </section>
      </div>
    </>
  );
}

function recoveryRefreshDelay(recovery: RunStatus["recovery"]): number | false {
  if (recovery === undefined || recovery.requiresRetry) return false;
  const next = Math.min(
    Date.parse(recovery.nextRetryAt ?? recovery.automaticUntil),
    Date.parse(recovery.automaticUntil),
  );
  // The deadline can pass with no Run event or changed snapshot. Keep a
  // bounded interval until the Server reports that recovery has ended.
  return Number.isFinite(next) ? Math.max(2000, next - Date.now()) : 2000;
}

export function RunDetailRoute() {
  const api = usePublicAPI();
  const { runId = "" } = useParams();
  const valid = RUN_ID_PATTERN.test(runId);
  const query = useQuery({
    queryKey: queryKeys.runs.detail(runId),
    queryFn: () => getRun(api, runId),
    enabled: valid,
    refetchInterval: (current) =>
      recoveryRefreshDelay(current.state.data?.recovery),
  });
  useDocumentTitle(
    query.data === undefined ? "Run" : `${query.data.workflow} · Run`,
  );
  if (!valid) {
    return (
      <div className="runs-page">
        <section className="runs-surface runs-run-empty">
          <ErrorNotice error={new Error("Run route is invalid")} />
          <Link to="/runs">Return to Runs</Link>
        </section>
      </div>
    );
  }
  const refresh = (
    <RefreshButton
      className="runs-icon-button"
      isFetching={query.isFetching}
      onRefresh={() => void query.refetch()}
    />
  );
  const run = query.data;
  return (
    <div className="runs-page">
      <article className="runs-surface runs-run" aria-labelledby="run-title">
        {run === undefined ? (
          <>
            <header className="runs-run-header">
              <RunBreadcrumb runId={runId} terminal={false} />
              <div className="runs-run-titles">
                <p className="runs-label runs-run-kind">Workflow Run</p>
                <h1 className="runs-run-title" id="run-title">
                  Run details
                </h1>
              </div>
              <div className="runs-run-actions">{refresh}</div>
            </header>
            <div className="runs-run-body">
              {query.error === null ? (
                <p className="runs-loading" aria-live="polite">
                  Loading Run…
                </p>
              ) : (
                <ErrorNotice
                  error={query.error}
                  context="Could not load this Run"
                  onRetry={() => void query.refetch()}
                  retryPending={query.isFetching}
                />
              )}
            </div>
          </>
        ) : (
          <LoadedRunDetail
            key={run.runId}
            run={run}
            snapshotVersion={query.dataUpdatedAt}
            refresh={refresh}
            staleError={query.error}
            onRetry={() => void query.refetch()}
            retryPending={query.isFetching}
          />
        )}
      </article>
    </div>
  );
}
