import "./runs.css";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import { Link, useLocation, useParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  isTerminalRunState,
  RUN_ID_PATTERN,
  type RunStatus,
} from "../../api/runs";
import { useDocumentTitle } from "../../app/document-title";
import { ErrorNotice } from "../../app/error-notice";
import {
  compactId,
  formatElapsed,
  formatTimestamp,
  plural,
} from "../../app/format";
import { catalogReturnState } from "../../app/navigation";
import { QueryView } from "../../app/query-view";
import { RefreshButton } from "../../app/refresh-button";
import {
  DetailHeader,
  Facts,
  IdChip,
  StatusGlyph,
  type StatusTone,
} from "../../ui";
import { useRunActions } from "./actions";
import { RunArtifactLibrary } from "./artifacts";
import {
  type RunDisclosureProps,
  RunSection,
  RunStateChip,
} from "./components";
import { parseWorkflowIdentity } from "./output-model";
import { RunOutputGallery, RunOutputPublications } from "./outputs";
import { runStatusQuery } from "./status";
import {
  LiveAttempts,
  type LiveStatus,
  RunBindings,
  RunCancellationRecord,
  RunExecutionMetrics,
  RunMetadataLabels,
  RunRuntimeConfiguration,
} from "./technical";
import { deriveRunTriage } from "./triage";
import { RunTriageSummary } from "./triage-summary";

const LIVE_TONES: Record<LiveStatus["connection"], StatusTone> = {
  live: "success",
  connecting: "progress",
  reconnecting: "warning",
  resyncing: "warning",
  error: "blocked",
  unavailable: "neutral",
};

function sameLiveStatus(left: LiveStatus, right: LiveStatus): boolean {
  return (
    left.connection === right.connection &&
    left.error === right.error &&
    left.resyncReason === right.resyncReason
  );
}

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

/**
 * The Run page header. Its breadcrumb is "Completed › run_0123…": the Runs
 * view the Run belongs to, or the page the user opened it from (with that
 * page's own return state). The rail already marks Runs as the destination,
 * so the trail does not repeat it.
 */
function RunHeader({
  runId,
  terminal,
  title,
  titleRef,
  status,
  actions,
  meta,
}: {
  runId: string;
  terminal: boolean;
  title: ReactNode;
  titleRef?: RefObject<HTMLHeadingElement | null>;
  status?: ReactNode;
  actions: ReactNode;
  meta?: ReactNode;
}) {
  const { state } = useLocation();
  const origin = catalogReturnState(
    state,
    terminal
      ? { returnTo: "/runs?view=completed", returnLabel: "Completed" }
      : { returnTo: "/runs", returnLabel: "Queue" },
  );
  return (
    <DetailHeader
      breadcrumb={[
        {
          label: origin.returnLabel,
          to: origin.returnTo,
          state: origin.returnState,
        },
        {
          label: (
            <span className="ui-mono" title={runId}>
              {compactId(runId)}
            </span>
          ),
        },
      ]}
      titleAs="h1"
      titleRef={titleRef}
      title={<span id="run-title">{title}</span>}
      eyebrow="Workflow Run"
      status={status}
      actions={actions}
      meta={meta}
    />
  );
}

/** "router-analysis@1" with the version quieter. */
function RunTitle({ workflow }: { workflow: string }) {
  const identity = parseWorkflowIdentity(workflow);
  return identity === undefined ? (
    workflow
  ) : (
    <>
      {identity.name}
      <span className="runs-run-version">@{identity.version}</span>
    </>
  );
}

function RunFacts({ run }: { run: RunStatus }) {
  const durationMs = deriveRunTriage(run).durationMs;
  const time = (value: string | undefined) =>
    value === undefined ? undefined : formatTimestamp(value);
  return (
    <Facts
      className="runs-facts run-metadata"
      items={[
        {
          term: "Run ID",
          value: <IdChip value={run.runId} label="Run ID" />,
        },
        {
          term: "Workflow",
          value: (
            <IdChip
              value={run.workflow}
              display={run.workflow}
              label="workflow version"
            />
          ),
        },
        {
          term: "Project",
          value:
            run.projectId === undefined ? undefined : (
              <Link to={`/projects/${encodeURIComponent(run.projectId)}`}>
                <code>{run.projectId}</code>
              </Link>
            ),
        },
        { term: "Created", value: time(run.createdAt) },
        { term: "Started", value: time(run.startedAt) },
        { term: "Updated", value: time(run.updatedAt) },
        { term: "Finished", value: time(run.finishedAt) },
        {
          term: "Duration",
          value:
            durationMs === undefined ? undefined : formatElapsed(durationMs),
        },
      ]}
    />
  );
}

/**
 * The header of a loaded Run: title, state, the actions with their
 * confirmations, facts and the event stream's state.
 */
function LoadedRunHeader({
  run,
  live,
  refresh,
  heading,
  onDone,
}: {
  run: RunStatus;
  live: LiveStatus;
  refresh: ReactNode;
  heading: RefObject<HTMLHeadingElement | null>;
  onDone: (announcement: string) => void;
}) {
  const terminal = isTerminalRunState(run.state);
  // The Dialog returns focus to its trigger first; the trigger is usually
  // gone with the new state, so the page heading takes focus instead.
  const focusHeading = useCallback(() => {
    window.setTimeout(() => heading.current?.focus(), 0);
  }, [heading]);
  const actions = useRunActions(
    run,
    (message) => {
      onDone(message);
      focusHeading();
    },
    focusHeading,
  );
  return (
    <>
      <RunHeader
        runId={run.runId}
        terminal={terminal}
        title={<RunTitle workflow={run.workflow} />}
        titleRef={heading}
        status={<RunStateChip state={run.state} size="md" />}
        actions={
          <>
            {actions.controls}
            {refresh}
          </>
        }
        meta={
          <>
            <RunFacts run={run} />
            <LiveStatusLine status={live} terminal={terminal} />
          </>
        }
      />
      {actions.notices}
      {actions.overlays}
    </>
  );
}

/**
 * The body of a loaded Run: triage, results, publications and Technical
 * details. It is mounted with the first snapshot, whose state decides which
 * sections start open; choices persist across live remounts because the
 * page is keyed by Run ID.
 */
function RunBody({
  run,
  snapshotVersion,
  liveError,
  onLiveStatus,
}: {
  run: RunStatus;
  snapshotVersion: number;
  liveError: string | undefined;
  onLiveStatus: (status: LiveStatus) => void;
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
  // Heavy sections stay open while a Run is active or failed (diagnostics
  // matter) and start collapsed once it ended otherwise; reference material
  // always starts collapsed.
  const [disclosures, setDisclosures] = useState(() => {
    const open = !isTerminalRunState(run.state) || run.state === "failed";
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

  return (
    <>
      {liveError === undefined ? null : (
        <div className="notice notice-warning" role="alert">
          <strong>{liveError}</strong>
          <p>Use Refresh to reload the Run.</p>
        </div>
      )}
      <RunTriageSummary run={run} triage={triage} onOpenAttempt={openAttempt} />
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
            <p className="ui-caption">
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
              onLiveStatus={onLiveStatus}
            />
          </RunSection>
          <RunMetadataLabels run={run} {...disclosure("labels")} />
          <RunBindings run={run} {...disclosure("bindings")} />
          <RunRuntimeConfiguration run={run} {...disclosure("runtime")} />
          <RunArtifactLibrary runId={run.runId} {...disclosure("artifacts")} />
          {run.cancellation === undefined ? null : (
            <RunCancellationRecord
              cancellation={run.cancellation}
              {...disclosure("cancellation")}
            />
          )}
        </div>
      </section>
    </>
  );
}

/** One Run, from its first read until the user leaves it. */
function RunPage({ runId }: { runId: string }) {
  const api = usePublicAPI();
  const query = useQuery(runStatusQuery(api, runId));
  useDocumentTitle(
    query.data === undefined ? "Run" : `${query.data.workflow} · Run`,
  );
  const heading = useRef<HTMLHeadingElement>(null);
  const [announcement, setAnnouncement] = useState("");
  const [reported, setReported] = useState<LiveStatus>();
  const reportLive = useCallback((next: LiveStatus) => {
    setReported((current) =>
      current !== undefined && sameLiveStatus(current, next) ? current : next,
    );
  }, []);
  const run = query.data;
  // Until the attempts report the stream's state, it is about to connect.
  const live: LiveStatus = reported ?? {
    connection: run?.eventCursor === undefined ? "unavailable" : "connecting",
  };
  const refresh = (
    <RefreshButton
      className="runs-icon-button"
      size="sm"
      isFetching={query.isFetching}
      onRefresh={() => void query.refetch()}
    />
  );
  return (
    <div className="runs-page">
      <article className="runs-surface runs-run" aria-labelledby="run-title">
        {run === undefined ? (
          <RunHeader
            runId={runId}
            terminal={false}
            title="Run details"
            actions={refresh}
          />
        ) : (
          <LoadedRunHeader
            run={run}
            live={live}
            refresh={refresh}
            heading={heading}
            onDone={setAnnouncement}
          />
        )}
        <p className="ui-visually-hidden" role="status">
          {announcement}
        </p>
        <div className="runs-run-body">
          <QueryView
            query={query}
            errorContext="Could not load this Run"
            loading={
              <p className="ui-caption" aria-live="polite">
                Loading Run…
              </p>
            }
          >
            {(loaded) => (
              <RunBody
                run={loaded}
                snapshotVersion={query.dataUpdatedAt}
                liveError={live.error}
                onLiveStatus={reportLive}
              />
            )}
          </QueryView>
        </div>
      </article>
    </div>
  );
}

export function RunDetailRoute() {
  const { runId = "" } = useParams();
  if (!RUN_ID_PATTERN.test(runId)) {
    return <InvalidRunRoute />;
  }
  return <RunPage key={runId} runId={runId} />;
}

function InvalidRunRoute() {
  useDocumentTitle("Run");
  return (
    <div className="runs-page">
      <section className="runs-surface runs-run-empty">
        <ErrorNotice error={new Error("Run route is invalid")} />
        <Link to="/runs">Return to Runs</Link>
      </section>
    </div>
  );
}
