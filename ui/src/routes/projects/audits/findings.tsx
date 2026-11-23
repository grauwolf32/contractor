import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { useId } from "react";
import { useParams, useSearchParams } from "react-router";
import { collectAuditPages } from "../../../api/audit-collections";
import {
  findingNeedsReview,
  findingSeverity,
  listProjectFindings,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { ErrorNotice } from "../../../app/error-notice";
import { FINDING_STATE_LABELS } from "../../../app/vocabulary";
import { Button, EmptyState, FilterChips, ListSection } from "../../../ui";
import { IssueRow } from "../../issues/issue-row";
import {
  issueHref,
  matchesIssueFilters,
  readIssueFilters,
} from "../../issues/links";
import { ProjectSectionActions } from "../navigation";
import {
  FINDING_STATES,
  SEVERITY_OPTIONS,
  DECISION_OPTIONS,
  isVerdictFilter,
} from "./finding-options";
import { LoadMoreControl } from "./load-more";
import "../../issues/issues.css";

/** One Project list, including discoveries from ordinary Runs and deleted checks.
 * Each explicit load reads at most five pages; refresh starts a new snapshot.
 */
export function ProjectFindingsRoute({ facts = false }: { facts?: boolean }) {
  const { projectId = "" } = useParams();
  const api = usePublicAPI();
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const headingId = useId();
  const searchId = useId();
  const severityId = useId();
  const decisionId = useId();
  const queryKey = ["project-findings", projectId, "list"];
  const findings = useInfiniteQuery({
    queryKey,
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam }) =>
      collectAuditPages(
        (cursor) =>
          listProjectFindings(
            api,
            projectId,
            cursor === undefined ? {} : { cursor },
          ),
        pageParam === undefined ? {} : { cursor: pageParam },
      ),
    getNextPageParam: (last) => (last.truncated ? last.nextCursor : undefined),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    retry: false,
  });
  const rows = findings.data?.pages.flatMap((page) => page.items) ?? [];
  const filters = readIssueFilters(
    new URLSearchParams({
      ...Object.fromEntries(params),
      state: params.get("state") ?? "all",
      ...(facts ? { severity: "informational" } : {}),
    }),
  );
  const search = (params.get("q") ?? "").trim().toLocaleLowerCase();
  const rawVerdict = params.get("verdict") ?? "";
  const verdict = isVerdictFilter(rawVerdict) ? rawVerdict : undefined;
  const matching = rows.filter(
    (row) =>
      (search === "" ||
        [row.title, row.subjectKey, row.findingId]
          .join(" ")
          .toLocaleLowerCase()
          .includes(search)) &&
      (filters.severity === undefined ||
        findingSeverity(row) === filters.severity) &&
      (verdict === undefined ||
        (verdict === "needs_review"
          ? findingNeedsReview(row)
          : row.analystVerdict === verdict)),
  );
  const visible = matching
    .filter((row) => matchesIssueFilters(row, filters))
    .sort(
      (a, b) =>
        Number(a.analystVerdict !== null) - Number(b.analystVerdict !== null) ||
        b.createdAt.localeCompare(a.createdAt) ||
        a.findingId.localeCompare(b.findingId),
    );
  const chips = [
    {
      value: "unassessed" as const,
      label: "Not assessed",
      count: matching.filter((row) => row.state === null).length,
    },
    ...FINDING_STATES.map((value) => ({
      value,
      label: FINDING_STATE_LABELS[value].label,
      count: matching.filter((row) => row.state === value).length,
    })),
    { value: "all" as const, label: "All", count: matching.length },
  ];
  function change(name: string, value: string) {
    setParams(
      (current) => {
        const next = new URLSearchParams(current);
        if (value === "" || value === "all") next.delete(name);
        else next.set(name, value);
        return next;
      },
      { replace: true },
    );
  }
  const refresh = () => {
    void client.resetQueries({ queryKey, exact: true });
  };
  return (
    <section className="route-page issues-project" aria-labelledby={headingId}>
      <ProjectSectionActions>
        <Button size="sm" pending={findings.isFetching} onClick={refresh}>
          Refresh
        </Button>
      </ProjectSectionActions>
      <header className="issues-section-head">
        <h3 id={headingId} className="issues-section-title">
          {facts ? "Facts" : "Possible issues"}
        </h3>
        <p className="ui-caption">
          {facts
            ? "Informational observations from either publishing tool. Sources retain their own claims and checks; classification does not verify a statement."
            : "One record per problem in this project, including discoveries from ordinary Runs."}
        </p>
      </header>
      <div
        className="issues-section-filters"
        role="group"
        aria-label={facts ? "Fact filters" : "Possible issue filters"}
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
              {facts ? "Search facts" : "Search possible issues"}
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
              disabled={facts}
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
          {facts
            ? " Facts use informational severity and do not need review. Other severities are available in Issues."
            : ""}
        </p>
      </div>
      <div className="issues-project-count">
        <span role="status">
          {facts
            ? `${visible.length} facts shown · ${rows.length} records loaded`
            : `${visible.length} of ${rows.length} possible issues loaded`}
        </span>
        {params.size === 0 ? null : (
          <Button size="xs" onClick={() => setParams({}, { replace: true })}>
            Clear filters
          </Button>
        )}
      </div>
      {findings.isPending ? (
        <p role="status">
          {facts ? "Loading facts…" : "Loading possible issues…"}
        </p>
      ) : null}
      {findings.error === null ? null : (
        <ErrorNotice error={findings.error} onRetry={refresh} />
      )}
      {!findings.isPending &&
      findings.error === null &&
      visible.length === 0 ? (
        <EmptyState
          title={
            facts
              ? "No facts in the loaded records"
              : rows.length === 0
                ? "No possible issues yet"
                : "No matching possible issues"
          }
        >
          <p>
            {facts && findings.hasNextPage
              ? "Load more records to extend this view."
              : rows.length === 0
                ? "Published problems appear here."
                : "Try another search or clear the filters."}
          </p>
        </EmptyState>
      ) : null}
      {visible.length === 0 ? null : (
        <ListSection aria-label={facts ? "Facts" : "Possible issues"}>
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
        shown={rows.length}
        noun={facts ? "records" : "possible issues"}
        truncated={findings.hasNextPage}
        loading={findings.isFetchingNextPage}
        error={findings.isFetchNextPageError ? findings.error : null}
        onLoadMore={() => {
          void findings.fetchNextPage();
        }}
        label={facts ? "Load more facts" : "Load more possible issues"}
      />
    </section>
  );
}
