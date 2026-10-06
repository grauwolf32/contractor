import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import { getOperationsSnapshot } from "../../api/operations";
import { queryKeys } from "../../api/query-keys";
import { Icon } from "../../app/icon";
import { RecordedTime } from "../../app/recorded-time";
import { checkStateLabel, reviewKindLabel } from "../../app/vocabulary";
import { useSession } from "../../auth/session";
import {
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  ProgressSegments,
  StatusGlyph,
  TechnicalDetails,
  type ListNavigationContainerProps,
  type StatusTone,
} from "../../ui";
import { weaknessReferences } from "../decisions/text";
import { describeStopReason } from "../projects/audits/stop-reason";
import { deriveRunTriage } from "../runs/triage";
import { useWorkflowInventory } from "../workflows/inventory";
import type { InboxData } from "./data";
import {
  inboxSearch,
  runTime,
  type InboxRow,
  type InboxRowOf,
  type InboxSectionId,
} from "./model";
import {
  checkKind,
  checkName,
  checkProgress,
  checkTypeLabel,
  excerpt,
  FAILED_RUNS_PATH,
  gapsTitle,
  primaryOutput,
  reviewSubjectItem,
  reviewTitle,
  SECTIONS,
  SUCCEEDED_RUNS_PATH,
} from "./present";

const UNAVAILABLE = "Some of this could not be loaded. Try again above.";

function rowLink(row: InboxRow) {
  return { pathname: "/", search: inboxSearch(row.ref) };
}

/** The status word of a row's meta line: the glyph is decorative. */
function State({ children }: { children: ReactNode }) {
  return <strong className="inbox-row-state">{children}</strong>;
}

function Project({ name }: { name: string }) {
  return <span className="inbox-row-project">{name}</span>;
}

interface RowProps<T extends InboxRow["type"]> {
  row: InboxRowOf<T>;
  selected: boolean;
  data: InboxData;
}

function IssueRow({ row, selected }: RowProps<"issue">) {
  const { project, audit, finding } = row.issue;
  const document = finding.firstProposal.document;
  const weakness = weaknessReferences(document)[0];
  return (
    <ListRow
      to={rowLink(row)}
      selected={selected}
      glyph={<StatusGlyph tone="review" />}
      title={document.title}
      meta={[
        <Project key="project" name={project.name} />,
        checkTypeLabel(audit),
        weakness === undefined ? (
          <RecordedTime key="time" value={finding.createdAt} />
        ) : (
          weakness.requirement_id
        ),
      ]}
    />
  );
}

function ReviewRow({ row, selected, data }: RowProps<"review">) {
  const { project, audit, review } = row.decision;
  const item = reviewSubjectItem(row.decision, data.items.byCheck);
  return (
    <ListRow
      to={rowLink(row)}
      selected={selected}
      glyph={<StatusGlyph tone="review" />}
      title={reviewTitle(row.decision, item, reviewKindLabel(review.kind))}
      meta={[
        item === undefined ? undefined : (
          <Project key="project" name={project.name} />
        ),
        checkTypeLabel(audit),
        <RecordedTime key="time" value={review.createdAt} />,
      ]}
    />
  );
}

const CHECK_GLYPHS: Readonly<
  Record<InboxRowOf<"check">["reason"], StatusTone>
> = {
  paused: "warning",
  failed: "blocked",
  gaps: "partial",
  running: "progress",
  finished: "done",
};

function CheckRow({ row, selected }: RowProps<"check">) {
  const { project, audit } = row.check;
  const glyph = <StatusGlyph tone={CHECK_GLYPHS[row.reason]} />;
  const kind = checkKind(row.check);
  const state = checkStateLabel(audit.state).label;
  switch (row.reason) {
    case "paused":
    case "failed": {
      const stop = describeStopReason(audit);
      const reason =
        stop === null
          ? undefined
          : (stop.label ??
            (stop.message.trim() === ""
              ? undefined
              : excerpt(stop.message, 80)));
      return (
        <ListRow
          to={rowLink(row)}
          selected={selected}
          glyph={glyph}
          title={checkName(row.check)}
          meta={[
            <State key="state">{state}</State>,
            reason,
            <RecordedTime
              key="time"
              value={
                row.reason === "paused"
                  ? (audit.pausedAt ?? audit.updatedAt)
                  : (audit.finishedAt ?? audit.updatedAt)
              }
            />,
          ]}
        />
      );
    }
    case "gaps":
      return (
        <ListRow
          to={rowLink(row)}
          selected={selected}
          glyph={glyph}
          title={gapsTitle(row.workspace?.gaps ?? 0, kind)}
          meta={[
            <Project key="project" name={project.name} />,
            checkTypeLabel(audit),
            state,
          ]}
        />
      );
    case "running": {
      const progress =
        row.workspace === undefined
          ? undefined
          : checkProgress(row.workspace, kind);
      return (
        <ListRow
          to={rowLink(row)}
          selected={selected}
          glyph={glyph}
          title={checkName(row.check)}
          meta={[
            progress?.summary,
            audit.state === "finalizing" || progress === undefined
              ? state
              : undefined,
          ]}
        >
          {progress === undefined || progress.total === 0 ? null : (
            <div className="inbox-row-progress">
              <ProgressSegments
                size="sm"
                label={progress.label}
                segments={progress.segments}
              />
            </div>
          )}
        </ListRow>
      );
    }
    case "finished":
      return (
        <ListRow
          to={rowLink(row)}
          selected={selected}
          glyph={glyph}
          title={`${project.name} check finished`}
          meta={[
            checkTypeLabel(audit),
            <RecordedTime
              key="time"
              value={audit.finishedAt ?? audit.updatedAt}
            />,
          ]}
        />
      );
  }
}

function ReportRow({ row, selected }: RowProps<"report">) {
  const { project, audit } = row.report;
  return (
    <ListRow
      to={rowLink(row)}
      selected={selected}
      glyph={<StatusGlyph tone="done" />}
      title={`${project.name} report is ready`}
      meta={[
        checkTypeLabel(audit),
        <RecordedTime key="time" value={audit.finishedAt ?? audit.updatedAt} />,
      ]}
    />
  );
}

/** The failure cause in the Server's words, or why it is not shown. */
function failureCause(row: InboxRowOf<"run">, data: InboxData) {
  if (row.status === undefined)
    return data.runStatuses.failed.has(row.run.runId)
      ? "The cause could not be loaded"
      : undefined;
  const issue = deriveRunTriage(row.status).issue;
  return issue === undefined
    ? "No cause was reported"
    : excerpt(issue.message, 120);
}

function RunRow({ row, selected, data }: RowProps<"run">) {
  const { run } = row;
  const project =
    run.projectId === undefined
      ? undefined
      : data.projectNames.get(run.projectId);
  let tone: StatusTone;
  let detail: ReactNode[];
  let note: string | undefined;
  if (row.reason === "model") {
    tone = "warning";
    detail = [<State key="state">Needs a model retry</State>];
  } else if (row.reason === "failed") {
    tone = "blocked";
    detail = [<State key="state">Failed</State>];
    note = failureCause(row, data);
  } else {
    tone = "done";
    const primary = primaryOutput(
      run,
      row.status,
      data.runStatuses.failed.has(run.runId),
      data.workflowOutputs,
    );
    detail = [
      primary.state === "present"
        ? `Primary result: ${primary.slot}`
        : primary.state === "missing"
          ? `Primary result ${primary.slot} is missing`
          : primary.state === "none"
            ? "No primary result declared"
            : undefined,
    ];
  }
  return (
    <ListRow
      to={rowLink(row)}
      selected={selected}
      glyph={<StatusGlyph tone={tone} />}
      title={run.workflow}
      clamp={1}
      meta={[
        ...detail,
        project,
        <RecordedTime key="time" value={runTime(run)} />,
      ]}
    >
      {note === undefined ? null : <p className="inbox-row-note">{note}</p>}
    </ListRow>
  );
}

function InboxListRow({
  row,
  selectedKey,
  data,
}: {
  row: InboxRow;
  selectedKey: string | undefined;
  data: InboxData;
}) {
  const selected = row.key === selectedKey;
  switch (row.type) {
    case "issue":
      return <IssueRow row={row} selected={selected} data={data} />;
    case "review":
      return <ReviewRow row={row} selected={selected} data={data} />;
    case "check":
      return <CheckRow row={row} selected={selected} data={data} />;
    case "report":
      return <ReportRow row={row} selected={selected} data={data} />;
    case "run":
      return <RunRow row={row} selected={selected} data={data} />;
  }
}

/** A note or link under a section's rows. */
function More({ children }: { children: ReactNode }) {
  return <li className="inbox-more">{children}</li>;
}

function Section({
  id,
  rows,
  selectedKey,
  data,
  empty,
  children,
}: {
  id: InboxSectionId;
  rows: readonly InboxRow[];
  selectedKey: string | undefined;
  data: InboxData;
  /** Nothing to list, not even the rows in `children`. */
  empty: boolean;
  /** Rows that are not items, notes and links after the rows. */
  children?: ReactNode;
}) {
  const section = SECTIONS[id];
  const pending = data.pending[id];
  return (
    <ListSection
      title={section.title}
      count={pending && rows.length === 0 ? undefined : rows.length}
      aside={section.aside}
    >
      {rows.map((row) => (
        <InboxListRow
          key={`${row.section}:${row.key}`}
          row={row}
          selectedKey={selectedKey}
          data={data}
        />
      ))}
      {!empty ? null : (
        <li className="inbox-section-empty">
          {pending
            ? "Loading…"
            : data.unavailable[id]
              ? UNAVAILABLE
              : section.empty}
        </li>
      )}
      {children}
    </ListSection>
  );
}

function plural(count: number, one: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`;
}

function DecideMore({ data }: { data: InboxData }) {
  const truncated =
    data.issues.truncatedAuditIds.length > 0 ||
    data.decisions.truncatedAuditIds.length > 0;
  if (!truncated) return null;
  return (
    <More>
      Some checks have more to decide than the Inbox lists.{" "}
      <Link to="/issues">Open Issues</Link>
    </More>
  );
}

function UnblockMore({ data }: { data: InboxData }) {
  const { recent, query } = data.failedRuns;
  if (recent.shown.length === 0) return null;
  return (
    <More>
      {recent.hidden > 0
        ? `${plural(recent.hidden, "more failed Run", "more failed Runs")} in the last 7 days. `
        : // The first page holds the newest Runs; a full one may hide more.
          query.data?.page.hasMore === true
          ? "Older failed Runs are not listed here. "
          : null}
      <Link to={FAILED_RUNS_PATH}>See all failed Runs</Link>
    </More>
  );
}

function ReadyMore({ data }: { data: InboxData }) {
  const { model } = data;
  const runs = data.succeededRuns.recent.hidden;
  return (
    <>
      {model.hiddenReports === 0 ? null : (
        <More>
          {plural(model.hiddenReports, "more report", "more reports")}.{" "}
          <Link to="/reports">See all reports</Link>
        </More>
      )}
      {model.hiddenFinishedChecks === 0 ? null : (
        <More>
          {plural(
            model.hiddenFinishedChecks,
            "more finished check",
            "more finished checks",
          )}
          . <Link to="/checks">See all checks</Link>
        </More>
      )}
      {runs === 0 ? null : (
        <More>
          {plural(runs, "more finished Run", "more finished Runs")}.{" "}
          <Link to={SUCCEEDED_RUNS_PATH}>See all finished Runs</Link>
        </More>
      )}
    </>
  );
}

function ActiveRunsRow({ data }: { data: InboxData }) {
  const page = data.activeRuns.data;
  if (page === undefined || page.items.length === 0) return null;
  const single = page.items.length === 1 && !page.page.hasMore;
  return (
    <ListRow
      to="/runs"
      glyph={<StatusGlyph tone="progress" />}
      title={
        single ? "1 Run in progress" : `${data.activeRunCount} Runs in progress`
      }
      meta="Follow them in Runs"
    />
  );
}

/**
 * Counts that were Home tiles: active and failed Runs, idle Runtime slots
 * and published Workflow versions. Read only while the disclosure is open.
 */
function RunsAndCapacity({ data }: { data: InboxData }) {
  const marker = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const details = marker.current?.closest("details");
    if (details === null || details === undefined) return undefined;
    const update = () => setOpen(details.open);
    details.addEventListener("toggle", update);
    return () => details.removeEventListener("toggle", update);
  }, []);
  return <div ref={marker}>{open ? <CapacityCounts data={data} /> : null}</div>;
}

function CapacityCounts({ data }: { data: InboxData }) {
  const api = usePublicAPI();
  const { session } = useSession();
  const operations =
    session?.principal.capabilities.includes("operations") === true;
  const snapshot = useQuery({
    queryKey: queryKeys.operations.snapshot,
    queryFn: () => getOperationsSnapshot(api),
    enabled: operations,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const workflows = useWorkflowInventory();
  const idle = snapshot.data?.runtimeAgents.filter(
    (agent) => agent.slotState === "idle",
  ).length;
  const failed = data.failedRuns.recent;
  return (
    <>
      <dl className="inbox-facts" data-size="sm">
        <div>
          <dt>Active Runs</dt>
          <dd>
            {data.activeRunCount ?? "—"} · <Link to="/runs">Open Runs</Link>
          </dd>
        </div>
        <div>
          <dt>Failed Runs, last 7 days</dt>
          <dd>
            {data.failedRuns.query.data === undefined
              ? "—"
              : failed.shown.length + failed.hidden}{" "}
            · <Link to={FAILED_RUNS_PATH}>See all</Link>
          </dd>
        </div>
        <div>
          <dt>Idle Runtime slots</dt>
          <dd>
            {!operations ? (
              "Needs operations access"
            ) : snapshot.data === undefined ? (
              snapshot.isError ? (
                "Could not be loaded"
              ) : (
                "Loading…"
              )
            ) : (
              <>
                {idle} of {snapshot.data.runtimeAgents.length} ·{" "}
                <Link to="/operations/runtime-agents">Runtime agents</Link>
              </>
            )}
          </dd>
        </div>
        <div>
          <dt>Published Workflow versions</dt>
          <dd>
            {workflows.data?.length ??
              (workflows.isError ? "Could not be loaded" : "Loading…")}{" "}
            · <Link to="/catalog/workflows">Workflows</Link>
          </dd>
        </div>
      </dl>
      <p className="inbox-quiet-note">
        An idle slot does not guarantee room for a particular Run.
      </p>
    </>
  );
}

export interface InboxListProps {
  data: InboxData;
  subtitle: string;
  selectedKey: string | undefined;
  containerProps: ListNavigationContainerProps;
}

/** The list pane: title, notices, the four sections and key hints. */
export function InboxList({
  data,
  subtitle,
  selectedKey,
  containerProps,
}: InboxListProps) {
  const { model } = data;
  const [retrying, setRetrying] = useState(false);
  const paused = data.queue.data?.paused === true;
  const activeRuns = data.activeRuns.data?.items.length ?? 0;
  function retry() {
    setRetrying(true);
    void data.retry().finally(() => setRetrying(false));
  }
  return (
    <ListPane
      title="Inbox"
      subtitle={subtitle}
      actions={
        // The lists refresh every 20 seconds; this reads them now.
        <button
          type="button"
          className="ui-btn inbox-refresh"
          data-size="xs"
          data-variant="ghost"
          aria-label="Refresh"
          title="Refresh"
          aria-busy={retrying || undefined}
          data-busy={retrying || undefined}
          disabled={retrying}
          onClick={retry}
        >
          <Icon name="refresh" />
        </button>
      }
      footer={
        <>
          <span className="inbox-keys">
            <Kbd>J</Kbd>
            <Kbd>K</Kbd> move
          </span>
          <span className="inbox-keys">
            <Kbd>Enter</Kbd> open
          </span>
          <span className="inbox-keys">
            <Kbd>C</Kbd>
            <Kbd>R</Kbd>
            <Kbd>E</Kbd> decide in place
          </span>
        </>
      }
    >
      {paused ? (
        <div className="inbox-notice" data-tone="warning" role="status">
          <StatusGlyph tone="warning" />
          <p>
            Queue admission is paused. New Runs wait until you resume it in
            Runs. <Link to="/runs">Open Runs</Link>
          </p>
        </div>
      ) : null}
      {data.incomplete ? (
        <div className="inbox-notice" role="status">
          <StatusGlyph tone="info" />
          <p>
            Some projects, checks or Runs could not be read, so the Inbox may be
            incomplete.{" "}
            <button
              type="button"
              className="ui-btn"
              data-size="xs"
              disabled={retrying}
              onClick={retry}
            >
              {retrying ? "Retrying…" : "Retry"}
            </button>
          </p>
        </div>
      ) : null}
      {data.checks.truncated ? (
        <p className="inbox-quiet-note">
          The Inbox reads your first 50 projects and the newest 50 checks of
          each. Checks, Issues and Reports list the rest.
        </p>
      ) : null}
      <div className="inbox-sections" {...containerProps}>
        <Section
          id="decide"
          rows={model.decide}
          selectedKey={selectedKey}
          data={data}
          empty={model.decide.length === 0}
        >
          <DecideMore data={data} />
        </Section>
        <Section
          id="unblock"
          rows={model.unblock}
          selectedKey={selectedKey}
          data={data}
          empty={model.unblock.length === 0}
        >
          <UnblockMore data={data} />
        </Section>
        <Section
          id="ready"
          rows={model.ready}
          selectedKey={selectedKey}
          data={data}
          empty={model.ready.length === 0}
        >
          <ReadyMore data={data} />
        </Section>
        <Section
          id="running"
          rows={model.running}
          selectedKey={selectedKey}
          data={data}
          empty={model.running.length === 0 && activeRuns === 0}
        >
          <ActiveRunsRow data={data} />
        </Section>
      </div>
      <TechnicalDetails
        summary="Runs and capacity"
        description="Run counts, idle Runtime slots and published Workflows."
      >
        <RunsAndCapacity data={data} />
      </TechnicalDetails>
    </ListPane>
  );
}
