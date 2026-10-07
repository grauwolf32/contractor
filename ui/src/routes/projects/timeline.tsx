import { type ReactNode, useId } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { Audit, AuditFinding, AuditState } from "../../api/audits";
import type { Project } from "../../api/projects";
import type { RunSummary, WorkflowRunState } from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import type { StatusTone } from "../../app/status-tone";
import { checkStateLabel } from "../../app/vocabulary";
import {
  type ActivityEntry,
  ActivityLog,
  EmptyState,
  FilterChips,
  StatusChip,
} from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { auditProfileLabel } from "./audits/labels";
import { materialKind, materialKindLabel } from "./material-kinds";
import type { CheckIssues } from "./overview-data";
import { RunPrimaryResult } from "./overview-runs";
import { projectPath } from "./project-sections";
import { runStateLabel } from "./run-state";
import { TIMELINE_FILTERS, type TimelineFilter } from "./timeline-filter";

/** How many events the timeline shows. */
const TIMELINE_LENGTH = 12;

interface TimelineEvent {
  id: string;
  kind: Exclude<TimelineFilter, "all"> | "project";
  /** ISO timestamp. */
  time: string;
  tone: StatusTone;
  title?: ReactNode;
  text?: ReactNode;
}

const CHECK_PHRASES: Readonly<Record<AuditState, string>> = {
  draft: "is a draft and has not started",
  active: "is running",
  waiting_review: "is waiting for you",
  paused: "is paused",
  finalizing: "is finishing",
  cancelling: "is stopping",
  completed: "finished",
  cancelled: "was stopped",
  failed: "failed",
  deleting: "is being deleted",
};

const RUN_PHRASES: Readonly<Record<WorkflowRunState, string>> = {
  initializing: "is initializing",
  pending: "is pending",
  waiting: "is waiting",
  running: "is running",
  cancelling: "is cancelling",
  succeeded: "succeeded",
  failed: "failed",
  cancelled: "was cancelled",
};

const HTTP_OPERATION =
  /^(get|put|post|delete|options|head|patch|trace)\s+(\/\S*)$/i;

function stopReason(audit: Audit): string | undefined {
  const reason = audit.stopReason;
  if (reason === undefined) return undefined;
  return reason.code === "deadline_exhausted"
    ? "time limit reached"
    : reason.message;
}

function checkEvent(
  audit: Audit,
  projectId: string,
  toReview: number,
): TimelineEvent {
  const reason = stopReason(audit);
  const phrase = Object.hasOwn(CHECK_PHRASES, audit.state)
    ? CHECK_PHRASES[audit.state]
    : `is ${checkStateLabel(audit.state).label.toLowerCase()}`;
  return {
    id: `check:${audit.auditId}`,
    kind: "checks",
    time:
      audit.finishedAt ??
      (audit.state === "draft" ? audit.createdAt : audit.updatedAt),
    tone: checkStateLabel(audit.state).tone,
    title: (
      <ContextLink
        returnLabel="Project Overview"
        to={`${projectPath(projectId, "audits")}/${encodeURIComponent(audit.auditId)}`}
      >
        {auditProfileLabel(audit)}
      </ContextLink>
    ),
    text: (
      <>
        {phrase}
        {reason === undefined ? "" : `: ${reason}`}.
        {toReview > 0
          ? ` ${toReview} ${toReview === 1 ? "possible issue needs" : "possible issues need"} review.`
          : null}
      </>
    ),
  };
}

function issueEvent(finding: AuditFinding): TimelineEvent {
  const document = finding.firstProposal.document;
  const operation =
    document.subject === null
      ? null
      : HTTP_OPERATION.exec(document.subject.key.trim());
  return {
    id: `issue:${finding.auditId}/${finding.findingId}`,
    kind: "issues",
    time: finding.createdAt,
    tone: "review",
    title: "Possible issue found",
    text: (
      <>
        {operation === null ? null : (
          <>
            on{" "}
            <span className="projects-mono">
              {operation[1]!.toUpperCase()} {operation[2]}
            </span>
            {": "}
          </>
        )}
        <ContextLink
          returnLabel="Project Overview"
          to={`/issues/${encodeURIComponent(finding.auditId)}/${encodeURIComponent(finding.findingId)}`}
        >
          {document.title}
        </ContextLink>{" "}
        <StatusChip tone="review" size="sm">
          Needs review
        </StatusChip>
      </>
    ),
  };
}

function runEvent(run: RunSummary, projectId: string): TimelineEvent {
  const phrase = Object.hasOwn(RUN_PHRASES, run.state)
    ? RUN_PHRASES[run.state]
    : runStateLabel(run.state).label.toLowerCase();
  return {
    id: `run:${run.runId}`,
    kind: "runs",
    time: run.finishedAt ?? run.updatedAt,
    tone: runStateLabel(run.state).tone,
    title: (
      <ContextLink
        returnLabel="Project Overview"
        to={`/runs/${encodeURIComponent(run.runId)}`}
      >
        {run.workflow}
      </ContextLink>
    ),
    text: (
      <>
        {phrase}.
        {run.state === "succeeded" ? (
          <RunPrimaryResult summary={run} projectId={projectId} />
        ) : null}
      </>
    ),
  };
}

function materialEvent(
  material: ArtifactMetadata,
  projectId: string,
): TimelineEvent {
  const { namespace, name } = material.artifact;
  return {
    id: `material:${namespace}/${name}`,
    kind: "materials",
    time: material.createdAt,
    tone: "neutral",
    title: materialKindLabel(materialKind(material)),
    text: (
      <>
        <ContextLink
          returnLabel="Project Overview"
          to={artifactDetailPath(
            { kind: "project", id: projectId },
            {
              namespace,
              name,
            },
          )}
        >
          {namespace}/{name}
        </ContextLink>{" "}
        saved.
      </>
    ),
  };
}

function timestamp(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

const dayFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
});
const datedFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
});

function startOfDay(value: Date): number {
  return new Date(
    value.getFullYear(),
    value.getMonth(),
    value.getDate(),
  ).valueOf();
}

/** "Today", "Yesterday", "Oct 4" or "Oct 4, 2025" in local time. */
function dayLabel(time: string, now: Date): string {
  const date = new Date(time);
  if (Number.isNaN(date.valueOf())) return "Earlier";
  const days = Math.round((startOfDay(now) - startOfDay(date)) / 86_400_000);
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  return date.getFullYear() === now.getFullYear()
    ? dayFormat.format(date)
    : datedFormat.format(date);
}

export interface ProjectTimelineProps {
  project: Project;
  checks: readonly Audit[];
  issues: readonly CheckIssues[];
  runs: readonly RunSummary[];
  materials: readonly ArtifactMetadata[];
  filter: TimelineFilter;
  onFilter: (filter: TimelineFilter) => void;
}

/**
 * Recent events of the project, newest first and grouped by day: checks,
 * possible issues that need review, Runs started outside checks, materials
 * and the project's creation. It is built from the overview's bounded reads,
 * so it shows recent activity, not the whole history.
 */
export function ProjectTimeline({
  project,
  checks,
  issues,
  runs,
  materials,
  filter,
  onFilter,
}: ProjectTimelineProps) {
  const heading = useId();
  const projectId = project.projectId;
  const toReview = new Map(
    issues.map((check) => [check.audit.auditId, check.total]),
  );
  const events: TimelineEvent[] = [
    ...checks.map((audit) =>
      checkEvent(audit, projectId, toReview.get(audit.auditId) ?? 0),
    ),
    ...issues.flatMap((check) => check.items.map(issueEvent)),
    // A check's own Runs are told by the check.
    ...runs
      .filter((run) => !("audit.id" in run.labels))
      .map((run) => runEvent(run, projectId)),
    ...materials.map((material) => materialEvent(material, projectId)),
    {
      id: "project",
      kind: "project",
      time: project.createdAt,
      tone: "neutral",
      title: "Project created",
    },
  ];
  const shown = events
    .filter((event) => filter === "all" || event.kind === filter)
    .sort(
      (left, right) =>
        timestamp(right.time) - timestamp(left.time) ||
        left.id.localeCompare(right.id),
    )
    .slice(0, TIMELINE_LENGTH);
  const now = new Date();
  const days: { label: string; entries: ActivityEntry[] }[] = [];
  for (const event of shown) {
    const label = dayLabel(event.time, now);
    let day = days.at(-1);
    if (day === undefined || day.label !== label) {
      day = { label, entries: [] };
      days.push(day);
    }
    day.entries.push({
      id: event.id,
      time: event.time,
      tone: event.tone,
      title: event.title,
      text: event.text,
    });
  }
  const filterLabel =
    filter === "all"
      ? "events"
      : TIMELINE_FILTERS.find(
          (option) => option.value === filter,
        )?.label.toLowerCase();

  return (
    <section className="projects-timeline" aria-labelledby={heading}>
      <div className="projects-section-heading">
        <h3 id={heading}>Timeline</h3>
        <FilterChips
          label="Show in timeline"
          options={TIMELINE_FILTERS}
          value={filter}
          onChange={onFilter}
        />
      </div>
      {days.length === 0 ? (
        <EmptyState title={`No recent ${filterLabel ?? "events"}`}>
          The timeline shows the most recent checks, possible issues, Runs and
          materials of this project.
        </EmptyState>
      ) : (
        <ol role="list" className="projects-timeline-days">
          {days.map((day) => (
            <li key={day.label} className="projects-timeline-day">
              <h4 className="projects-timeline-date">{day.label}</h4>
              <ActivityLog
                aria-label={`Timeline, ${day.label}`}
                entries={day.entries}
              />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
