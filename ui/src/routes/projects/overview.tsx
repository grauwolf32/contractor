import { useId, type ReactNode } from "react";
import { Link, useLocation, useSearchParams } from "react-router";

import { AUDIT_PAGE_SIZE, type Audit, type AuditState } from "../../api/audits";
import type { Project } from "../../api/projects";
import type { RunSummary } from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import { checkStateLabel } from "../../app/vocabulary";
import { ButtonLink, StatusChip, StatusGlyph } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { auditProfileLabel } from "./audits/labels";
import { CheckComposer } from "./check-composer";
import { ProjectSectionActions } from "./navigation";
import {
  type MaterialsSample,
  type PossibleIssuesToReview,
  useMaterialsSample,
  usePossibleIssuesToReview,
  useProjectChecks,
  useRecentRuns,
  useSuccessfulRuns,
  useWaitingChecks,
} from "./overview-data";
import { RecentResults, RecentRuns } from "./overview-runs";
import {
  LIVE_TARGET_ANCHOR,
  projectPath,
  targetSheetState,
} from "./project-sections";
import { ProjectTimeline } from "./timeline";
import {
  parseTimelineFilter,
  serializeTimelineFilter,
  type TimelineFilter,
} from "./timeline-filter";
import { plural } from "../../app/format";
import {
  MATERIAL_KIND_LABELS,
  MATERIAL_KIND_ORDER,
  materialFormat,
  materialKindLabel,
  materialKindOf,
} from "../artifacts/kinds";
import { LIVE_AUDIT_STATES } from "../../api/states";

const NO_CHECKS: Audit[] = [];

/**
 * Check states besides running that the "Running checks" cell names: still
 * live, but not running (contract §3 labels: Finishing, Stopping, Waiting for
 * you).
 */
const OTHER_LIVE_STATES: readonly AuditState[] = LIVE_AUDIT_STATES.filter(
  (state) => state !== "active",
);

/** Material chips shown before "All materials". */
const CHIP_LIMIT = 4;

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** Exact sampled materials and the optional live target. */
function MaterialChips({
  project,
  materials,
}: {
  project: Project;
  materials: MaterialsSample;
}) {
  const projectId = project.projectId;
  const location = useLocation();
  const shown = materials.items.slice(0, CHIP_LIMIT);
  const hidden = materials.items.length - shown.length;
  const target = project.httpTarget;
  const settings = `${projectPath(projectId, "settings")}#${LIVE_TARGET_ANCHOR}`;
  return (
    <section className="projects-materials" aria-label="Project materials">
      <div className="projects-section-heading">
        <h3>Materials</h3>
        <Link to={`${projectPath(projectId, "artifacts")}?add=artifact`}>
          <Icon name="plus" size={13} />
          Add material
        </Link>
      </div>
      <ul role="list" aria-label="Materials" className="projects-material-list">
        {materials.isPending ? (
          <li className="ui-caption" role="status">
            Loading materials…
          </li>
        ) : null}
        {shown.map((item) => {
          const exact = `${item.artifact.namespace}/${item.artifact.name}@${item.artifact.revision}`;
          return (
            <li key={exact}>
              <ContextLink
                className="projects-material-item"
                returnLabel="Project Overview"
                title={exact}
                to={artifactDetailPath(
                  { kind: "project", id: projectId },
                  item.artifact,
                )}
              >
                <strong>
                  {materialKindLabel(item)} · {item.artifact.name} ·{" "}
                  {materialFormat(item)}
                </strong>
                <code>{exact}</code>
              </ContextLink>
            </li>
          );
        })}
        {hidden > 0 || materials.more ? (
          <li>
            <Link to={projectPath(projectId, "artifacts")}>
              {hidden > 0 && !materials.more
                ? `${hidden} more`
                : "All materials"}
            </Link>
          </li>
        ) : null}
      </ul>
      {materials.error === null ? null : (
        <ErrorNotice
          error={materials.error}
          context="Could not load materials"
          onRetry={materials.refetch}
          retryPending={materials.isFetching}
        />
      )}
      <div className="projects-material-context">
        {target === undefined ? (
          <>
            <span>
              Live target not configured ·{" "}
              <Link
                to={settings}
                state={targetSheetState(location.state)}
                aria-label="Add a live target"
              >
                Add live target
              </Link>
            </span>
            <span>
              Source and API-based checks remain available without a live
              target.
            </span>
          </>
        ) : (
          <Link to={settings} title={target.url}>
            <Icon name="target" size={14} /> Live target · {hostOf(target.url)}
          </Link>
        )}
      </div>
    </section>
  );
}

function Cell({
  label,
  value,
  caption,
}: {
  label: string;
  value: ReactNode;
  caption?: ReactNode;
}) {
  return (
    <div className="projects-glance-cell">
      <dt>{label}</dt>
      <dd>
        <span className="projects-glance-value">{value}</span>
        {caption === undefined || caption === null ? null : (
          <span className="projects-glance-caption">{caption}</span>
        )}
      </dd>
    </div>
  );
}

interface LastActivity {
  time: string;
  what: string;
}

function lastActivity(
  project: Project,
  checks: readonly Audit[],
  runs: readonly RunSummary[],
  materials: MaterialsSample,
): LastActivity {
  const candidates: LastActivity[] = [
    { time: project.updatedAt, what: "Project updated" },
    ...checks.map((audit) => ({
      time: audit.updatedAt,
      what: `${auditProfileLabel(audit)} updated`,
    })),
    ...runs.map((run) => ({
      time: run.updatedAt,
      what: `Run of ${run.workflow} updated`,
    })),
    ...materials.items.map((item) => ({
      time: item.createdAt,
      what: `${item.artifact.namespace}/${item.artifact.name} saved`,
    })),
  ];
  return candidates.reduce((newest, candidate) =>
    Date.parse(candidate.time) > Date.parse(newest.time) ? candidate : newest,
  );
}

/** Qualified summary counts from the existing bounded reads. */
function GlanceStrip({
  project,
  checks,
  checksTruncated,
  checksState,
  issues,
  runs,
  materials,
}: {
  project: Project;
  checks: readonly Audit[];
  checksTruncated: boolean;
  checksState: "pending" | "error" | "ready";
  issues: PossibleIssuesToReview;
  runs: readonly RunSummary[];
  materials: MaterialsSample;
}) {
  const heading = useId();
  const projectId = project.projectId;
  const inState = (state: AuditState) =>
    checks.filter((audit) => audit.state === state).length;
  const running = inState("active");
  // Checks between running and ended, by their state's own label.
  const others = OTHER_LIVE_STATES.flatMap((state) => {
    const count = inState(state);
    return count === 0
      ? []
      : [`${count} ${checkStateLabel(state).label.toLowerCase()}`];
  });
  const newest = `In the ${AUDIT_PAGE_SIZE} newest checks`;

  let issuesValue: ReactNode;
  let issuesCaption: ReactNode;
  if (
    checksState === "pending" ||
    (checksState === "ready" && !issues.settled)
  ) {
    issuesValue = "Counting…";
  } else if (checksState === "error") {
    issuesValue = "Unavailable";
  } else if (issues.total > 0) {
    issuesValue = (
      <Link
        to={`/issues?${new URLSearchParams({ project: projectId }).toString()}`}
      >
        <StatusGlyph tone="review" />
        {issues.partial ? "At least " : ""}
        {issues.total} to review
      </Link>
    );
  } else {
    issuesValue = issues.partial ? "Partly unavailable" : "None to review";
  }
  if (checksState === "ready") {
    issuesCaption = issues.partial
      ? "Some possible issues could not be read"
      : checksTruncated
        ? newest
        : undefined;
  }

  let runningValue: ReactNode;
  if (checksState === "pending") runningValue = "Loading…";
  else if (checksState === "error") runningValue = "Unavailable";
  else if (running > 0)
    runningValue = (
      <Link to={projectPath(projectId, "audits")}>
        <StatusGlyph tone="progress" />
        {running} running
      </Link>
    );
  else runningValue = "None running";
  const runningCaption =
    checksState !== "ready"
      ? undefined
      : others.length > 0
        ? others.join(" · ")
        : checks.length === 0
          ? "No checks yet"
          : checksTruncated
            ? newest
            : plural(checks.length, "check", "checks") + " in total";

  const kinds = [
    ...new Set(materials.items.map((item) => materialKindOf(item))),
  ].sort(
    (left, right) =>
      MATERIAL_KIND_ORDER.indexOf(left) - MATERIAL_KIND_ORDER.indexOf(right),
  );
  let materialsValue: ReactNode;
  if (materials.isPending) materialsValue = "Loading…";
  else if (materials.error !== null) materialsValue = "Unavailable";
  else if (materials.items.length === 0) materialsValue = "No materials yet";
  else
    materialsValue = (
      <Link to={projectPath(projectId, "artifacts")}>
        {materials.more
          ? `${materials.items.length}+ materials`
          : plural(materials.items.length, "material", "materials")}
      </Link>
    );

  const last = lastActivity(project, checks, runs, materials);

  return (
    <section className="projects-glance" aria-labelledby={heading}>
      <div className="projects-section-heading">
        <h3 id={heading}>At a glance</h3>
      </div>
      <dl>
        <Cell
          label="Possible issues"
          value={issuesValue}
          caption={issuesCaption}
        />
        <Cell
          label="Running checks"
          value={runningValue}
          caption={runningCaption}
        />
        <Cell
          label="Materials"
          value={materialsValue}
          caption={
            kinds.map((kind) => MATERIAL_KIND_LABELS[kind]).join(", ") ||
            undefined
          }
        />
        <Cell
          label="Last activity"
          value={<RecordedTime value={last.time} />}
          caption={last.what}
        />
      </dl>
      <p className="projects-glance-note">
        Recent checks and sampled materials; limitations are shown with each
        count.
      </p>
    </section>
  );
}

/** Review needs from the same bounded reads used by the summary counts. */
function Attention({
  projectId,
  waiting,
  issues,
}: {
  projectId: string;
  waiting: ReturnType<typeof useWaitingChecks>;
  issues: PossibleIssuesToReview;
}) {
  const heading = useId();
  const items = waiting.data?.items ?? [];
  const proposed = issues.items.slice(0, 3);
  const failed = waiting.error !== null && waiting.data === undefined;
  if (items.length === 0 && proposed.length === 0 && !failed) return null;
  return (
    <section className="projects-attention" aria-labelledby={heading}>
      <div className="projects-section-heading">
        <h3 id={heading}>Needs your attention</h3>
        {waiting.data?.page.hasMore ? (
          <Link to={projectPath(projectId, "audits")}>All checks</Link>
        ) : null}
      </div>
      {failed ? (
        <ErrorNotice
          error={waiting.error}
          context="Could not check for decisions"
          onRetry={() => void waiting.refetch()}
          retryPending={waiting.isFetching}
        />
      ) : null}
      <ul role="list" className="projects-attention-list">
        {items.map((audit) => (
          <li className="projects-attention-check" key={audit.auditId}>
            <div className="projects-row-main">
              <strong>{auditProfileLabel(audit)}</strong>
              <span className="projects-row-meta">
                <StatusChip tone="review" size="sm">
                  Waiting for you
                </StatusChip>
                <span>
                  updated <RecordedTime value={audit.updatedAt} />
                </span>
              </span>
            </div>
            <ButtonLink
              size="sm"
              returnLabel="Project Overview"
              to={`${projectPath(projectId, "audits")}/${encodeURIComponent(audit.auditId)}/reviews?state=pending`}
            >
              Review decisions
            </ButtonLink>
          </li>
        ))}
        {proposed.map((finding) => (
          <li className="projects-attention-issue" key={finding.findingId}>
            <StatusChip tone="review" size="sm" glyph={false}>
              Possible issue
            </StatusChip>
            <ContextLink
              returnLabel="Project Overview"
              to={`/issues/${encodeURIComponent(projectId)}/${encodeURIComponent(finding.findingId)}`}
            >
              {finding.title}
            </ContextLink>
            <span>Needs review</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * Objective and review needs lead; results, Runs and Timeline follow with
 * materials and qualified counts as supporting context. Every read is bounded;
 * counts say when they cover only the newest checks.
 */
export function ProjectOverview({ project }: { project: Project }) {
  const projectId = project.projectId;
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const timelineFilter = parseTimelineFilter(searchParams.get("timeline"));
  const checks = useProjectChecks(projectId);
  const checkItems = checks.data?.items ?? NO_CHECKS;
  const waiting = useWaitingChecks(projectId);
  const issues = usePossibleIssuesToReview(projectId);
  const runs = useRecentRuns(projectId);
  const results = useSuccessfulRuns(projectId);
  const materials = useMaterialsSample(projectId);
  const checksState =
    checks.data !== undefined
      ? "ready"
      : checks.error !== null
        ? "error"
        : "pending";

  function setTimelineFilter(filter: TimelineFilter) {
    const next = new URLSearchParams(searchParams);
    if (filter === "all") next.delete("timeline");
    else next.set("timeline", serializeTimelineFilter(filter));
    setSearchParams(next, {
      replace: true,
      preventScrollReset: true,
      state: location.state,
    });
  }

  const fetching =
    checks.isFetching ||
    waiting.isFetching ||
    runs.isFetching ||
    results.isFetching ||
    materials.isFetching;

  return (
    <div className="projects-overview">
      <ProjectSectionActions>
        <RefreshButton
          size="sm"
          isFetching={fetching}
          onRefresh={() => {
            void issues.refetch();
            void checks.refetch();
            void waiting.refetch();
            void runs.refetch();
            void results.refetch();
            materials.refetch();
          }}
        />
      </ProjectSectionActions>
      <CheckComposer project={project} materials={materials.items} />
      <Attention projectId={projectId} waiting={waiting} issues={issues} />
      {checks.error !== null && checks.data === undefined ? (
        <ErrorNotice
          error={checks.error}
          context="Could not load checks"
          onRetry={() => void checks.refetch()}
          retryPending={checks.isFetching}
        />
      ) : null}
      <div className="projects-overview-columns">
        <div className="projects-overview-main">
          <RecentResults projectId={projectId} results={results} />
          <RecentRuns projectId={projectId} runs={runs} />
          {checksState === "pending" ? (
            <p className="loading-copy" role="status">
              Loading the timeline…
            </p>
          ) : (
            <ProjectTimeline
              project={project}
              checks={checkItems}
              issues={issues.items}
              runs={runs.data?.items ?? []}
              materials={materials.items}
              filter={timelineFilter}
              onFilter={setTimelineFilter}
            />
          )}
        </div>
        <aside
          className="projects-overview-support"
          aria-label="Project context"
        >
          <GlanceStrip
            project={project}
            checks={checkItems}
            checksTruncated={checks.data?.page.hasMore === true}
            checksState={checksState}
            issues={issues}
            runs={runs.data?.items ?? []}
            materials={materials}
          />
          <MaterialChips project={project} materials={materials} />
        </aside>
      </div>
    </div>
  );
}
