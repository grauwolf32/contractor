import { Link, useNavigate, useSearchParams } from "react-router";

import {
  useAllReports,
  useProjectsIndex,
  type CrossProjectReport,
} from "../../api/cross-project";
import type { Project } from "../../api/projects";
import { RecordedTime } from "../../app/recorded-time";
import { reportStatusLabel } from "../../app/vocabulary";
import {
  Button,
  EmptyState,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  StatusGlyph,
  shortenId,
  useListNavigation,
} from "../../ui";
import { auditProfileLabel } from "../../modules/checks/labels";
import {
  PROJECT_PARAM,
  PAGE_PARAM,
  reportPath,
  type ReportFilters,
} from "./filters";

import "./reports.css";

function subtitle(count: number, loading: boolean): string {
  if (loading) return "Loading reports…";
  return count === 0 ? "No reports yet." : `${count} ready.`;
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
        className="ui-control"
        data-size="sm"
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
  const { tone } = reportStatusLabel(entry.report.status);
  const objective = entry.audit.scope.objective?.trim();
  return (
    <ListRow
      to={reportPath(entry.audit.auditId, filters)}
      selected={selected}
      glyph={<StatusGlyph tone={tone} />}
      title={
        objective ||
        `${auditProfileLabel(entry.audit)} · ${shortenId(entry.audit.auditId)}`
      }
      meta={[
        <span key="project" className="reports-row-project">
          {entry.project.name}
        </span>,
        ...(objective
          ? [<span key="type">{auditProfileLabel(entry.audit)}</span>]
          : []),
        <RecordedTime
          key="finished"
          value={entry.audit.finishedAt ?? entry.audit.updatedAt}
        />,
      ]}
    />
  );
}

/**
 * The Reports list pane: reports of every listed project, filtered by project and page in the URL.
 */
export function ReportsList({
  selectedId,
  filters,
}: {
  selectedId: string | undefined;
  filters: ReportFilters;
}) {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const reports = useAllReports({
    projectId: filters.project,
    page: filters.page,
  });
  const index = useProjectsIndex();
  const rows = reports.reports;
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
    const next = new URLSearchParams(params);
    if (key !== PAGE_PARAM) next.delete(PAGE_PARAM);
    if (value === "" || (key === PAGE_PARAM && value === "0")) next.delete(key);
    else next.set(key, value);
    // Scope and page changes leave the old report before another can be downloaded.
    void navigate(
      { pathname: "/reports", search: next.toString() },
      { preventScrollReset: true },
    );
  }

  // Counts only once every read has settled; until then they would grow. A
  // list that could not be read has no counts: unknown is not zero.
  const indexError = reports.error;

  const paged = reports.page.count > 1;
  const pageStart = reports.page.index * reports.page.size + 1;
  const pageEnd = Math.min(
    pageStart + reports.page.size - 1,
    reports.page.totalChecks,
  );
  return (
    <ListPane
      title="Reports"
      subtitle={
        indexError === null
          ? paged && !reports.isPending
            ? `${rows.length} ready in this page of checks.`
            : subtitle(rows.length, reports.isPending)
          : undefined
      }
      toolbar={
        <div className="reports-filters">
          <ProjectFilter
            projects={index.projects}
            value={filters.project}
            onChange={(projectId) => setFilter(PROJECT_PARAM, projectId)}
          />
        </div>
      }
      footer={
        paged || rows.length > 1 ? (
          <>
            {paged ? (
              <div className="ui-stack">
                <p className="ui-caption">
                  Reports from checks {pageStart}–{pageEnd} of{" "}
                  {reports.page.totalChecks} loaded checks. Counts describe this
                  page.
                </p>
                <nav className="pagination" aria-label="Report pages">
                  <Button
                    size="sm"
                    disabled={reports.page.index === 0}
                    onClick={() =>
                      setFilter(PAGE_PARAM, String(reports.page.index - 1))
                    }
                  >
                    Previous
                  </Button>
                  <span className="ui-caption">
                    Page {reports.page.index + 1} of {reports.page.count}
                  </span>
                  <Button
                    size="sm"
                    disabled={!reports.page.hasNext}
                    onClick={() =>
                      setFilter(PAGE_PARAM, String(reports.page.index + 1))
                    }
                  >
                    Next
                  </Button>
                </nav>
              </div>
            ) : null}
            {rows.length > 1 ? (
              <span>
                <Kbd>J</Kbd> <Kbd>K</Kbd> move between reports
              </span>
            ) : null}
          </>
        ) : undefined
      }
    >
      {indexError !== null ? (
        <EmptyState
          title="Reports could not be loaded"
          action={
            <Button size="sm" onClick={() => void reports.refetch()}>
              Try again
            </Button>
          }
        >
          <p>{indexError.message}</p>
        </EmptyState>
      ) : (
        <>
          {reports.truncated ? (
            <div className="reports-list-note" role="status">
              <p>
                There are more checks than can be read at once, so this list of
                reports is incomplete.
              </p>
            </div>
          ) : null}
          {reports.partial ? (
            <div className="reports-list-note" role="status">
              <p>
                Some reports could not be loaded, so this list may be
                incomplete.
              </p>
              <Button size="xs" onClick={() => void reports.refetch()}>
                Try again
              </Button>
            </div>
          ) : null}
          {rows.length === 0 ? (
            reports.isPending ? (
              <p className="ui-caption reports-list-loading" role="status">
                Loading reports…
              </p>
            ) : (
              <EmptyState
                title={
                  paged
                    ? "No ready reports in these checks"
                    : filters.project === null
                      ? "No reports yet"
                      : "No reports in this project yet"
                }
                action={
                  filters.project === null ? (
                    <Link to="/checks">Go to checks</Link>
                  ) : undefined
                }
              >
                <p>
                  {paged
                    ? "Use Previous or Next to inspect reports from another page of checks."
                    : "A check writes its report when it finishes."}
                </p>
              </EmptyState>
            )
          ) : (
            <div {...containerProps}>
              <ListSection title="Ready" count={rows.length}>
                {rows.map((entry) => (
                  <ReportRow
                    key={entry.audit.auditId}
                    entry={entry}
                    filters={filters}
                    selected={entry.audit.auditId === selectedId}
                  />
                ))}
              </ListSection>
              {reports.isPending ? (
                <p className="ui-caption reports-list-loading" role="status">
                  Loading more reports…
                </p>
              ) : null}
            </div>
          )}
        </>
      )}
    </ListPane>
  );
}
