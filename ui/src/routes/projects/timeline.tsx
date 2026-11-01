import { type ReactNode } from "react";

import type { ArtifactMetadata } from "../../api/artifacts";
import type { Audit, ProjectFindingSummary, AuditState } from "../../api/audits";
import type { Project } from "../../api/projects";
import type { RunSummary } from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import type { StatusTone } from "../../app/status-tone";
import {
  checkStateLabel,
  runStateLabel,
  runStatePhrase,
} from "../../app/vocabulary";
import {
  type ActivityEntry,
  ActivityLog,
  EmptyState,
  TechnicalDetails,
  StatusChip,
} from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { auditProfileLabel } from "./audits/labels";
import type { CheckIssues } from "./overview-data";
import { RunPrimaryResult } from "./overview-runs";
import { projectPath } from "./project-sections";
import {
  TIMELINE_FILTERS,
  type TimelineFilter,
  type TimelineKind,
} from "./timeline-filter";
import { formatDay, plural } from "../../app/format";
import { materialKindLabel } from "../artifacts/kinds";

/** How many events the timeline shows. */
const TIMELINE_LENGTH = 12;

interface TimelineEvent {
  id: string;
  kind: TimelineKind | "project";
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
        {toReview > 0 ? (
          <>
            {" "}
            <StatusChip tone="review" size="sm" glyph={false}>
              {plural(toReview, "possible issue needs", "possible issues need")}{" "}
              review
            </StatusChip>
          </>
        ) : null}
      </>
    ),
  };
}

function issueEvent(finding: ProjectFindingSummary): TimelineEvent {
  return {
    id: `issue:${finding.projectId}/${finding.findingId}`,
    kind: "issues",
    time: finding.createdAt,
    tone: "review",
    title: "Possible issue found",
    text: (
      <>
        <ContextLink
          returnLabel="Project Overview"
          to={`/issues/${encodeURIComponent(finding.projectId)}/${encodeURIComponent(finding.findingId)}`}
        >
          {finding.title}
        </ContextLink>{" "}
        <StatusChip tone="review" size="sm" glyph={false}>
          Needs review
        </StatusChip>
      </>
    ),
  };
}

function runEvent(run: RunSummary, projectId: string): TimelineEvent {
  const phrase = runStatePhrase(run.state);
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
    title: materialKindLabel(material),
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

/** "Today", "Yesterday", "Oct 4" or "Oct 4, 2025" in local time. */
function dayLabel(time: string, now: Date): string {
  return formatDay(time, now) ?? "Earlier";
}

interface ProjectTimelineProps {
  project: Project;
  checks: readonly Audit[];
  issues: readonly ProjectFindingSummary[];
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
  const projectId = project.projectId;
  const events: TimelineEvent[] = [
    ...checks.map((audit) =>
      checkEvent(audit, projectId, 0),
    ),
    ...issues.map(issueEvent),
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
    .filter(
      (event) =>
        filter === "all" ||
        (event.kind !== "project" && filter.includes(event.kind)),
    )
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
  function toggle(kind: TimelineKind) {
    const selected = new Set(
      filter === "all" ? TIMELINE_FILTERS.map(({ value }) => value) : filter,
    );
    if (selected.has(kind)) selected.delete(kind);
    else selected.add(kind);
    onFilter(
      selected.size === TIMELINE_FILTERS.length
        ? "all"
        : TIMELINE_FILTERS.filter(({ value }) => selected.has(value)).map(
            ({ value }) => value,
          ),
    );
  }
  return (
    <section className="projects-timeline" aria-label="Timeline">
      <TechnicalDetails
        className="projects-timeline-disclosure"
        summary="Timeline"
        description={`${shown.length} recent entries`}
        defaultOpen={filter !== "all"}
      >
        <div
          className="projects-timeline-filters"
          role="group"
          aria-label="Show in timeline"
        >
          <span>Show in timeline</span>
          <label>
            <input
              type="checkbox"
              checked={filter === "all"}
              onChange={() => onFilter(filter === "all" ? [] : "all")}
            />
            All
          </label>
          {TIMELINE_FILTERS.map(({ value, label }) => (
            <label key={value}>
              <input
                type="checkbox"
                checked={filter === "all" || filter.includes(value)}
                onChange={() => toggle(value)}
              />
              {label}
            </label>
          ))}
        </div>
        {days.length === 0 ? (
          <EmptyState title="No recent events for these filters">
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
      </TechnicalDetails>
    </section>
  );
}
