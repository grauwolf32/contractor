import { useMemo } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";

import {
  CROSS_PROJECT_LIMITS,
  useAllReports,
  useProjectsIndex,
  type CrossProjectReport,
} from "../../api/cross-project";
import type { Project } from "../../api/projects";
import { RecordedTime } from "../../app/recorded-time";
import { reportStatusLabel } from "../../app/vocabulary";
import {
  EmptyState,
  FilterChips,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  StatusChip,
  StatusGlyph,
  useListNavigation,
} from "../../ui";
import { auditProfileLabel } from "../projects/audits/labels";
import {
  PROJECT_PARAM,
  reportPath,
  STATUS_PARAM,
  type ReportFilter,
  type ReportFilters,
} from "./filters";

import "./reports.css";

type ListedStatus = Exclude<ReportFilter, "all">;

// Sections in display order: what needs the user first.
const SECTIONS: readonly { status: ListedStatus }[] = [
  { status: "proposed" },
  { status: "ready" },
];

function countOf(reports: readonly CrossProjectReport[], status: string) {
  return reports.filter((entry) => entry.report.status === status).length;
}

function subtitle(
  counts: { proposed: number; ready: number },
  loading: boolean,
): string {
  if (counts.proposed + counts.ready === 0)
    return loading ? "Loading reports…" : "No reports yet.";
  const parts: string[] = [];
  if (counts.proposed > 0)
    parts.push(`${counts.proposed} waiting for acceptance`);
  if (counts.ready > 0) parts.push(`${counts.ready} ready`);
  return `${parts.join(", ")}.`;
}

function emptyCopy(filters: ReportFilters): { title: string; body: string } {
  const where = filters.project === null ? "" : " in this project";
  switch (filters.status) {
    case "proposed":
      return {
        title: `Nothing is waiting for acceptance${where}`,
        body: "Reports that need your acceptance appear here.",
      };
    case "ready":
      return {
        title: `No ready reports${where}`,
        body: "Accepted reports appear here.",
      };
    case "all":
      return {
        title: `No reports${where} yet`,
        body: "A check writes its report when it finishes.",
      };
  }
}

function byName(left: Project, right: Project): number {
  return left.name.localeCompare(right.name, "en", { sensitivity: "base" });
}

function ProjectFilter({
  projects,
  value,
  onChange,
}: {
  projects: readonly Project[];
  value: string | null;
  onChange: (projectId: string) => void;
}) {
  const options = [...projects].sort(byName);
  const unknown =
    value !== null && !options.some((project) => project.projectId === value);
  return (
    <label className="reports-project-filter">
      <span>Project</span>
      <select
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">All projects</option>
        {unknown ? <option value={value}>{value}</option> : null}
        {options.map((project) => (
          <option key={project.projectId} value={project.projectId}>
            {project.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function ReportRow({
  entry,
  filters,
  selected,
}: {
  entry: CrossProjectReport;
  filters: ReportFilters;
  selected: boolean;
}) {
  const { label, tone } = reportStatusLabel(entry.report.status);
  return (
    <ListRow
      to={reportPath(entry.audit.auditId, filters)}
      selected={selected}
      glyph={<StatusGlyph tone={tone} />}
      title={
        <>
          {auditProfileLabel(entry.audit)}{" "}
          <span className="reports-row-project">· {entry.project.name}</span>
        </>
      }
      meta={[
        <StatusChip key="status" tone={tone} size="sm" glyph={false}>
          {label}
        </StatusChip>,
        // The check's last update; the exact time is the hover title.
        <RecordedTime key="updated" value={entry.audit.updatedAt} />,
      ]}
    />
  );
}

/**
 * The Reports list pane: reports of every listed project, filtered by status
 * and project in the URL (`?status=proposed|ready&project=<projectId>`).
 * Reports waiting for acceptance come first.
 */
export function ReportsList({
  selectedId,
  filters,
}: {
  selectedId: string | undefined;
  filters: ReportFilters;
}) {
  const navigate = useNavigate();
  const [, setParams] = useSearchParams();
  const reports = useAllReports();
  const index = useProjectsIndex();
  const scoped = useMemo(
    () =>
      filters.project === null
        ? reports.reports
        : reports.reports.filter(
            (entry) => entry.project.projectId === filters.project,
          ),
    [filters.project, reports.reports],
  );
  const counts = {
    proposed: countOf(scoped, "proposed"),
    ready: countOf(scoped, "ready"),
  };
  const sections = SECTIONS.filter(
    (section) => filters.status === "all" || filters.status === section.status,
  )
    .map((section) => ({
      ...section,
      rows: scoped.filter((entry) => entry.report.status === section.status),
    }))
    .filter((section) => section.rows.length > 0);
  const rows = sections.flatMap((section) => section.rows);
  const { containerProps } = useListNavigation({
    count: rows.length,
    index: rows.findIndex((entry) => entry.audit.auditId === selectedId),
    onMove: (position) => {
      const entry = rows[position];
      if (entry !== undefined)
        void navigate(reportPath(entry.audit.auditId, filters));
    },
  });

  function setFilter(key: string, value: string) {
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        if (value === "" || value === "all") next.delete(key);
        else next.set(key, value);
        return next;
      },
      { replace: true, preventScrollReset: true },
    );
  }

  // Counts only once every read has settled; until then they would grow. A
  // list that could not be read has no counts: unknown is not zero.
  const indexError = reports.error;
  const settled = !reports.isPending && indexError === null;
  const empty = emptyCopy(filters);
  return (
    <ListPane
      title="Reports"
      subtitle={
        indexError === null ? subtitle(counts, reports.isPending) : undefined
      }
      toolbar={
        <div className="reports-filters">
          <FilterChips<ReportFilter>
            label="Filter by status"
            value={filters.status}
            onChange={(value) => setFilter(STATUS_PARAM, value)}
            options={[
              {
                value: "proposed",
                label: reportStatusLabel("proposed").label,
                count: settled ? counts.proposed : undefined,
              },
              {
                value: "ready",
                label: reportStatusLabel("ready").label,
                count: settled ? counts.ready : undefined,
              },
              {
                value: "all",
                label: "All",
                count: settled ? counts.proposed + counts.ready : undefined,
              },
            ]}
          />
          <ProjectFilter
            projects={index.projects}
            value={filters.project}
            onChange={(projectId) => setFilter(PROJECT_PARAM, projectId)}
          />
        </div>
      }
      footer={
        rows.length > 1 ? (
          <span>
            <Kbd>J</Kbd> <Kbd>K</Kbd> move between reports
          </span>
        ) : undefined
      }
    >
      {indexError !== null ? (
        <EmptyState
          title="Reports could not be loaded"
          action={
            <button
              type="button"
              className="ui-btn"
              data-size="sm"
              onClick={() => void reports.refetch()}
            >
              Try again
            </button>
          }
        >
          <p>{indexError.message}</p>
        </EmptyState>
      ) : (
        <>
          {reports.partial ? (
            <div className="reports-list-note" role="status">
              <p>
                Some reports could not be loaded, so this list may be
                incomplete.
              </p>
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                onClick={() => void reports.refetch()}
              >
                Try again
              </button>
            </div>
          ) : null}
          {rows.length === 0 ? (
            reports.isPending ? (
              <p className="reports-quiet reports-list-loading" role="status">
                Loading reports…
              </p>
            ) : (
              <EmptyState
                title={empty.title}
                action={
                  filters.status === "all" && filters.project === null ? (
                    <Link to="/checks">Go to checks</Link>
                  ) : undefined
                }
              >
                <p>{empty.body}</p>
              </EmptyState>
            )
          ) : (
            <div {...containerProps} className="reports-sections">
              {sections.map((section) => {
                const { label } = reportStatusLabel(section.status);
                return (
                  <ListSection
                    key={section.status}
                    title={label}
                    count={section.rows.length}
                  >
                    {section.rows.map((entry) => (
                      <ReportRow
                        key={entry.audit.auditId}
                        entry={entry}
                        filters={filters}
                        selected={entry.audit.auditId === selectedId}
                      />
                    ))}
                  </ListSection>
                );
              })}
              {reports.isPending ? (
                <p className="reports-quiet reports-list-loading" role="status">
                  Loading more reports…
                </p>
              ) : null}
            </div>
          )}
          {reports.truncated ? (
            <p className="reports-quiet reports-list-limit">
              This list reads the first {CROSS_PROJECT_LIMITS.projects} projects
              and the {CROSS_PROJECT_LIMITS.auditsPerProject} newest checks of
              each. Open a project for all of its checks.
            </p>
          ) : null}
        </>
      )}
    </ListPane>
  );
}
