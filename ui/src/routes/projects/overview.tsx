import { useId, type ReactNode } from "react";
import { Link, useLocation, useSearchParams } from "react-router";

import type { Audit, AuditState } from "../../api/audits";
import { CROSS_PROJECT_LIMITS } from "../../api/cross-project";
import type { Project } from "../../api/projects";
import type { RunSummary } from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice } from "../../app/error-notice";
import { Icon } from "../../app/icon";
import { RecordedTime } from "../../app/recorded-time";
import { RefreshButton } from "../../app/refresh-button";
import { checkStateLabel } from "../../app/vocabulary";
import { StatusGlyph } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { auditProfileLabel } from "./audits/labels";
import { CheckComposer } from "./check-composer";
import { MaterialKindIcon } from "./material-icon";
import {
  MATERIAL_KIND_ORDER,
  materialFormat,
  materialKind,
  materialKindLabel,
} from "./material-kinds";
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
import { parseTimelineFilter, type TimelineFilter } from "./timeline-filter";

const NO_CHECKS: Audit[] = [];

/**
 * Check states besides running that the "Running checks" cell names: still
 * live, but not running (contract §3 labels: Finishing, Stopping, Waiting for
 * you).
 */
const OTHER_LIVE_STATES: readonly AuditState[] = [
  "finalizing",
  "cancelling",
  "waiting_review",
];

/** Material chips shown before "All materials". */
const CHIP_LIMIT = 4;

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** Source code, API spec, … and the live target, with "Add material". */
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
    <div className="projects-materials">
      <ul role="list" aria-label="Materials" className="projects-chips">
        {materials.isPending ? (
          <li>
            <span className="projects-chip" data-variant="quiet" role="status">
              Loading materials…
            </span>
          </li>
        ) : null}
        {shown.map((item) => {
          const kind = materialKind(item);
          const { namespace, name } = item.artifact;
          return (
            <li key={`${namespace}/${name}`}>
              <ContextLink
                className="projects-chip"
                returnLabel="Project Overview"
                to={artifactDetailPath(
                  { kind: "project", id: projectId },
                  {
                    namespace,
                    name,
                  },
                )}
                title={`${namespace}/${name}`}
              >
                <span className="projects-chip-icon">
                  <MaterialKindIcon kind={kind} />
                </span>
                <span className="projects-chip-label">
                  {materialKindLabel(kind)}
                </span>
                <span className="projects-chip-detail">
                  {name} · {materialFormat(item)}
                </span>
              </ContextLink>
            </li>
          );
        })}
        {hidden > 0 || materials.more ? (
          <li>
            <Link
              className="projects-chip"
              data-variant="quiet"
              to={projectPath(projectId, "artifacts")}
            >
              {hidden > 0 && !materials.more
                ? `${hidden} more`
                : "All materials"}
            </Link>
          </li>
        ) : null}
        <li>
          {target === undefined ? (
            <span className="projects-chip" data-variant="missing">
              <span className="projects-chip-icon">
                <MaterialKindIcon kind="target" />
              </span>
              <span className="projects-chip-label">Live target</span>
              <span className="projects-chip-detail">not configured</span>
              {/* Opens Settings at the Live target with its sheet open. */}
              <Link
                to={settings}
                state={targetSheetState(location.state)}
                aria-label="Add a live target"
              >
                Add
              </Link>
            </span>
          ) : (
            <Link className="projects-chip" to={settings} title={target.url}>
              <span className="projects-chip-icon">
                <MaterialKindIcon kind="target" />
              </span>
              <span className="projects-chip-label">Live target</span>
              <span className="projects-chip-detail">{hostOf(target.url)}</span>
            </Link>
          )}
        </li>
        <li>
          <Link
            className="projects-chip"
            data-variant="add"
            to={`${projectPath(projectId, "artifacts")}?add=artifact`}
          >
            <Icon name="plus" />
            Add material
          </Link>
        </li>
      </ul>
      {materials.error === null ? null : (
        <ErrorNotice
          error={materials.error}
          context="Could not load materials"
          onRetry={materials.refetch}
          retryPending={materials.isFetching}
        />
      )}
    </div>
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

/** The four-cell "at a glance" strip. */
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
  const newest = `In the ${CROSS_PROJECT_LIMITS.auditsPerProject} newest checks`;

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
        to={`/issues?${new URLSearchParams({ state: "proposed", project: projectId }).toString()}`}
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
      ? "Some checks could not be read"
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
    ...new Set(materials.items.map((item) => materialKind(item))),
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
      <h3 className="ui-visually-hidden" id={heading}>
        At a glance
      </h3>
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
          caption={kinds.map(materialKindLabel).join(", ") || undefined}
        />
        <Cell
          label="Last activity"
          value={<RecordedTime value={last.time} />}
          caption={last.what}
        />
      </dl>
    </section>
  );
}

/** Checks waiting for the user's decision. */
function Attention({
  projectId,
  waiting,
}: {
  projectId: string;
  waiting: ReturnType<typeof useWaitingChecks>;
}) {
  const heading = useId();
  if (waiting.error !== null && waiting.data === undefined) {
    return (
      <ErrorNotice
        error={waiting.error}
        context="Could not check for decisions"
        onRetry={() => void waiting.refetch()}
        retryPending={waiting.isFetching}
      />
    );
  }
  const items = waiting.data?.items ?? [];
  if (items.length === 0) return null;
  return (
    <section
      className="projects-panel projects-attention"
      aria-labelledby={heading}
    >
      <div className="projects-section-heading">
        <h3 id={heading}>
          <StatusGlyph tone="review" />
          Needs your attention
        </h3>
        {waiting.data?.page.hasMore ? (
          <Link to={projectPath(projectId, "audits")}>All checks</Link>
        ) : null}
      </div>
      <ul role="list" className="projects-rows">
        {items.map((audit) => (
          <li className="projects-row" key={audit.auditId}>
            <StatusGlyph tone="review" />
            <div className="projects-row-main">
              <strong>{auditProfileLabel(audit)}</strong>
              <span className="projects-row-meta">
                <span>Waiting for you</span>
                <span>
                  updated <RecordedTime value={audit.updatedAt} />
                </span>
              </span>
            </div>
            <ContextLink
              className="ui-btn"
              data-size="sm"
              returnLabel="Project Overview"
              to={`${projectPath(projectId, "audits")}/${encodeURIComponent(audit.auditId)}/reviews?state=pending`}
            >
              Review decisions
            </ContextLink>
          </li>
        ))}
      </ul>
    </section>
  );
}

/**
 * The project Overview (V3B): materials, the "at a glance" strip, checks
 * waiting for a decision, the check composer, the timeline, recent results
 * and recent Runs. Every read is bounded (S06 "Project section navigation");
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
  const issues = usePossibleIssuesToReview(checkItems);
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
    else next.set("timeline", filter);
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
          isFetching={fetching}
          onRefresh={() => {
            void checks.refetch();
            void waiting.refetch();
            void runs.refetch();
            void results.refetch();
            materials.refetch();
          }}
        />
      </ProjectSectionActions>
      <MaterialChips project={project} materials={materials} />
      <GlanceStrip
        project={project}
        checks={checkItems}
        checksTruncated={checks.data?.page.hasMore === true}
        checksState={checksState}
        issues={issues}
        runs={runs.data?.items ?? []}
        materials={materials}
      />
      <Attention projectId={projectId} waiting={waiting} />
      <CheckComposer project={project} materials={materials.items} />
      {checks.error !== null && checks.data === undefined ? (
        <ErrorNotice
          error={checks.error}
          context="Could not load checks"
          onRetry={() => void checks.refetch()}
          retryPending={checks.isFetching}
        />
      ) : null}
      {checksState === "pending" ? (
        <p className="loading-copy" role="status">
          Loading the timeline…
        </p>
      ) : (
        <ProjectTimeline
          project={project}
          checks={checkItems}
          issues={issues.checks}
          runs={runs.data?.items ?? []}
          materials={materials.items}
          filter={timelineFilter}
          onFilter={setTimelineFilter}
        />
      )}
      <div className="projects-overview-columns">
        <RecentResults projectId={projectId} results={results} />
        <RecentRuns projectId={projectId} runs={runs} />
      </div>
    </div>
  );
}
