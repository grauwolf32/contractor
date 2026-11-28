import { useId } from "react";
import { useParams } from "react-router";
import { ErrorNotice } from "../../../app/error-notice";
import { Button, EmptyState, FilterChips, ListSection } from "../../../ui";
import { IssueRow } from "../../issues/issue-row";
import { issueHref } from "../../issues/links";
import { ProjectSectionActions } from "../navigation";
import { SEVERITY_OPTIONS, DECISION_OPTIONS } from "./finding-options";
import { useProjectFindingView } from "./project-finding-view";
import { LoadMoreControl } from "./load-more";
import "../../issues/issues.css";

export function ProjectFindingsRoute() {
  const { projectId = "" } = useParams();
  const {
    params,
    findings,
    filters,
    verdict,
    visible,
    chips,
    change,
    refresh,
    total,
    clear,
  } = useProjectFindingView(projectId, false);
  const headingId = useId(),
    searchId = useId(),
    severityId = useId(),
    decisionId = useId();
  return (
    <section className="route-page issues-project" aria-labelledby={headingId}>
      <ProjectSectionActions>
        <Button size="sm" pending={findings.isFetching} onClick={refresh}>
          Refresh
        </Button>
      </ProjectSectionActions>
      <header className="issues-section-head">
        <h3 id={headingId} className="issues-section-title">
          Possible issues
        </h3>
        <p className="ui-caption">
          One record per problem in this project, including discoveries from
          ordinary Runs.
        </p>
      </header>
      <div
        className="issues-section-filters"
        role="group"
        aria-label="Possible issue filters"
      >
        <FilterChips
          label="Filter by state"
          options={chips}
          value={filters.state}
          onChange={(value) => change("state", value)}
        />
        <div className="issues-filter-row">
          <span className="issues-search-field">
            <label htmlFor={searchId} className="issues-field-label">
              Search possible issues
            </label>
            <input
              id={searchId}
              type="search"
              className="issues-search"
              value={params.get("q") ?? ""}
              onChange={(e) => change("q", e.target.value)}
              placeholder="Title, endpoint or ID"
            />
          </span>
          <span className="issues-select-field">
            <label htmlFor={severityId} className="issues-field-label">
              Severity
            </label>
            <select
              id={severityId}
              className="issues-select"
              value={filters.severity ?? ""}
              onChange={(e) => change("severity", e.target.value)}
            >
              {SEVERITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </span>
          <span className="issues-select-field">
            <label htmlFor={decisionId} className="issues-field-label">
              Decision
            </label>
            <select
              id={decisionId}
              className="issues-select"
              value={verdict ?? ""}
              onChange={(e) => change("verdict", e.target.value)}
            >
              {DECISION_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </span>
        </div>
        <p className="ui-caption">
          Filters and counts describe loaded records. Severity is the
          analyst&apos;s rating when present, otherwise the source suggestion.
        </p>
      </div>
      <div className="issues-project-count">
        <span role="status">
          {visible.length} of {total} possible issues loaded
        </span>
        {params.size === 0 ? null : (
          <Button size="xs" onClick={clear}>
            Clear filters
          </Button>
        )}
      </div>
      {findings.isPending ? (
        <p role="status">Loading possible issues…</p>
      ) : null}
      {findings.error === null ? null : (
        <ErrorNotice error={findings.error} onRetry={refresh} />
      )}
      {!findings.isPending &&
      findings.error === null &&
      visible.length === 0 ? (
        <EmptyState
          title={
            total === 0
              ? "No possible issues yet"
              : "No matching possible issues"
          }
        >
          <p>
            {total === 0
              ? "Published problems appear here."
              : "Try another search or clear the filters."}
          </p>
        </EmptyState>
      ) : null}
      {visible.length === 0 ? null : (
        <ListSection aria-label="Possible issues">
          {visible.map((finding) => (
            <IssueRow
              key={finding.findingId}
              id={`finding-${finding.findingId}`}
              finding={finding}
              to={issueHref(finding, { ...filters, project: projectId })}
            />
          ))}
        </ListSection>
      )}
      <LoadMoreControl
        shown={total}
        noun="possible issues"
        truncated={findings.hasNextPage}
        loading={findings.isFetchingNextPage}
        error={findings.isFetchNextPageError ? findings.error : null}
        onLoadMore={() => {
          void findings.fetchNextPage();
        }}
        label="Load more possible issues"
      />
    </section>
  );
}
