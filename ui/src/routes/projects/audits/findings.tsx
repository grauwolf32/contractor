import { useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { Link, useLocation, useParams, useSearchParams } from "react-router";

import {
  collectAuditPages,
  type AuditCollection,
} from "../../../api/audit-collections";
import {
  auditPollInterval,
  listAuditFindings,
  listProjectAudits,
  type Audit,
  type AuditFinding,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { ErrorNotice } from "../../../app/error-notice";
import { QueryView } from "../../../app/query-view";
import { RefreshButton } from "../../../app/refresh-button";
import {
  FINDING_STATE_LABELS,
  VERDICT_LABELS,
  type VerdictValue,
} from "../../../app/vocabulary";
import {
  EmptyState,
  FilterChips,
  ListSection,
  shortenId,
  type FilterChipOption,
} from "../../../ui";
import { IssueRow } from "../../issues/issue-row";
import { issueHref, type StateFilter } from "../../issues/links";
import { ProjectSectionActions } from "../navigation";
import { useAuditCollection, useAuditCollections } from "./collections";
import {
  FINDING_STATES,
  SEVERITY_OPTIONS,
  isFindingSeverity,
  isFindingState,
} from "./finding-options";
import { auditProfileLabel } from "./labels";
import { LoadMoreControl } from "./load-more";
import { AuditAnchor } from "./shared";

import "./styles.css";
import "../../issues/issues.css";

/** Decision filter values: none yet, then every analyst verdict. */
const DECISIONS: readonly VerdictValue[] = [
  "unreviewed",
  ...(Object.keys(VERDICT_LABELS) as VerdictValue[]).filter(
    (value) => value !== "unreviewed",
  ),
];

function isDecision(value: string): value is VerdictValue {
  return (DECISIONS as readonly string[]).includes(value);
}

interface Row {
  audit: Audit;
  finding: AuditFinding;
}

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

/**
 * The project's Issues tab: possible issues of every check, not reviewed
 * first and then newest. Search, check, state, severity (the analyst's
 * rating only) and decision filters live in the query string (q, audit,
 * state, severity, verdict). Each row opens the possible issue on the
 * Issues destination, where it is decided.
 */
export function ProjectFindingsRoute() {
  const api = usePublicAPI();
  const { projectId = "" } = useParams();
  const [filters, setFilters] = useSearchParams();
  const location = useLocation();
  const headingId = useId();
  const searchId = useId();
  const checkId = useId();
  const severityId = useId();
  const decisionId = useId();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validProject,
  });
  const audits = useAuditCollection<AuditCollection<Audit>>({
    queryKey: queryKeys.projects.audits.inventory(projectId),
    loadBatch: (cursor) =>
      collectAuditPages(
        (pageCursor) =>
          listProjectAudits(api, {
            projectId,
            ...(pageCursor === undefined ? {} : { cursor: pageCursor }),
          }),
        cursor === undefined ? {} : { cursor },
      ),
    enabled: validProject && project.data?.kind === "project",
    // The inventory only changes on its own while some check is running.
    refetchInterval: (loaded) => auditPollInterval(loaded),
  });
  const sources = audits.items;
  const findings = useAuditCollections(sources, {
    id: (audit) => audit.auditId,
    queryKey: (audit) => queryKeys.audits.allFindings(audit.auditId),
    load: (audit) => (cursor) =>
      listAuditFindings(
        api,
        audit.auditId,
        cursor === undefined ? {} : { cursor },
      ),
    identity: (finding) => finding.findingId,
    refetchInterval: (audit) => auditPollInterval([audit]),
  });
  const query = (filters.get("q") ?? "").trim().toLocaleLowerCase();
  const selectedAudit = filters.get("audit") ?? "";
  const rawSeverity = filters.get("severity") ?? "";
  const severity = isFindingSeverity(rawSeverity) ? rawSeverity : undefined;
  const rawVerdict = filters.get("verdict") ?? "";
  const verdict = isDecision(rawVerdict) ? rawVerdict : undefined;
  const rawState = filters.get("state") ?? "";
  const state: StateFilter = isFindingState(rawState) ? rawState : "all";
  const rows: Row[] = sources.flatMap((audit, index) =>
    (findings.results[index]?.items ?? []).map((finding) => ({
      audit,
      finding,
    })),
  );
  // Every filter but the state, so each state chip counts what it would show.
  const unstated = rows.filter(({ audit, finding }) => {
    const document = finding.firstProposal.document;
    return (
      (selectedAudit === "" || audit.auditId === selectedAudit) &&
      (verdict === undefined ||
        (verdict === "unreviewed"
          ? finding.analystVerdict === undefined
          : finding.analystVerdict === verdict)) &&
      // The analyst's rating only; the AI's suggestion is never a severity.
      (severity === undefined || finding.analystSeverity === severity) &&
      (query === "" ||
        [
          document.title,
          document.description,
          document.subject?.key ?? "",
          finding.findingId,
          auditProfileLabel(audit),
        ]
          .join(" ")
          .toLocaleLowerCase()
          .includes(query))
    );
  });
  const visible = unstated
    .filter(({ finding }) => state === "all" || finding.state === state)
    .sort(
      (a, b) =>
        Number(a.finding.analystVerdict !== undefined) -
          Number(b.finding.analystVerdict !== undefined) ||
        b.finding.createdAt.localeCompare(a.finding.createdAt) ||
        a.finding.findingId.localeCompare(b.finding.findingId),
    );
  const chips: FilterChipOption<StateFilter>[] = [
    ...FINDING_STATES.map((value) => ({
      value,
      label: FINDING_STATE_LABELS[value].label,
      count: unstated.filter(({ finding }) => finding.state === value).length,
    })),
    { value: "all", label: "All", count: unstated.length },
  ];
  const loading = findings.results.some((result) => result.isPending);
  const refreshing =
    audits.isFetching || findings.results.some((result) => result.isFetching);
  const failed = sources.filter((_, index) => findings.results[index]?.isError);
  const moreFindings = findings.results.find(
    (result) => result.moreError !== null,
  );
  const filtered =
    query !== "" ||
    selectedAudit !== "" ||
    severity !== undefined ||
    verdict !== undefined ||
    state !== "all";

  function setFilter(name: string, value: string) {
    setFilters(
      (current) => {
        const next = new URLSearchParams(current);
        if (value === "") next.delete(name);
        else next.set(name, value);
        return next;
      },
      { replace: true, state: location.state },
    );
  }

  if (!validProject)
    return (
      <section className="route-page issues-project">
        <ErrorNotice error={new Error("This project link is invalid.")} />
        <Link to="/projects">Return to Projects</Link>
      </section>
    );

  const refreshControl = (
    <RefreshButton
      isFetching={refreshing}
      onRefresh={() =>
        void Promise.all([
          audits.refetch(),
          ...findings.results.map((result) => result.refetch()),
        ])
      }
    />
  );
  return (
    <section className="route-page issues-project" aria-labelledby={headingId}>
      <ProjectSectionActions>{refreshControl}</ProjectSectionActions>
      <header className="issues-section-head">
        <h3 id={headingId} className="issues-section-title">
          Possible issues
        </h3>
        <p className="issues-quiet">
          From every check of this project: not reviewed first, then newest.
        </p>
      </header>
      <QueryView
        query={project}
        loading={<p className="issues-quiet">Loading the project…</p>}
        onRetry={() => void project.refetch()}
      >
        {(data) =>
          data.kind !== "project" ? (
            <ErrorNotice
              error={
                new Error("Possible issues are available for projects only.")
              }
            />
          ) : audits.error !== null ? (
            <ErrorNotice error={audits.error} />
          ) : audits.isPending ? (
            <p className="issues-quiet" role="status">
              Loading checks…
            </p>
          ) : (
            <>
              <div
                className="issues-section-filters"
                role="group"
                aria-label="Possible issue filters"
              >
                <FilterChips
                  label="Filter by state"
                  options={chips}
                  value={state}
                  onChange={(value) =>
                    setFilter("state", value === "all" ? "" : value)
                  }
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
                      value={filters.get("q") ?? ""}
                      onChange={(event) => setFilter("q", event.target.value)}
                      placeholder="Title, description, endpoint or ID"
                    />
                  </span>
                  <span className="issues-select-field">
                    <label htmlFor={checkId} className="issues-field-label">
                      Check
                    </label>
                    <select
                      id={checkId}
                      className="issues-select"
                      value={selectedAudit}
                      onChange={(event) =>
                        setFilter("audit", event.target.value)
                      }
                    >
                      <option value="">All checks ({sources.length})</option>
                      {sources.map((audit) => (
                        <option key={audit.auditId} value={audit.auditId}>
                          {auditProfileLabel(audit)} ·{" "}
                          {shortenId(audit.auditId)}
                        </option>
                      ))}
                    </select>
                  </span>
                  <span className="issues-select-field">
                    <label htmlFor={severityId} className="issues-field-label">
                      Severity
                    </label>
                    <select
                      id={severityId}
                      className="issues-select"
                      aria-describedby={`${severityId}-hint`}
                      value={severity ?? ""}
                      onChange={(event) =>
                        setFilter("severity", event.target.value)
                      }
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
                      onChange={(event) =>
                        setFilter("verdict", event.target.value)
                      }
                    >
                      <option value="">All decisions</option>
                      {DECISIONS.map((value) => (
                        <option key={value} value={value}>
                          {VERDICT_LABELS[value].label}
                        </option>
                      ))}
                    </select>
                  </span>
                </div>
                <p id={`${severityId}-hint`} className="issues-quiet">
                  Severity is the analyst&apos;s rating; AI suggestions are not
                  filtered.
                </p>
              </div>
              <div className="issues-project-count">
                <span role="status">
                  {visible.length} of{" "}
                  {plural(rows.length, "possible issue", "possible issues")}
                  {loading || failed.length > 0 ? " loaded" : ""} ·{" "}
                  {plural(sources.length, "check", "checks")}
                </span>
                {filtered ? (
                  <button
                    className="ui-btn"
                    data-size="xs"
                    type="button"
                    onClick={() =>
                      setFilters({}, { replace: true, state: location.state })
                    }
                  >
                    Clear filters
                  </button>
                ) : null}
              </div>
              {loading ? (
                <p className="issues-quiet" role="status">
                  Loading possible issues from every check…
                </p>
              ) : null}
              {failed.map((audit) => {
                const index = sources.indexOf(audit);
                return (
                  <div
                    className="issues-notice"
                    data-tone="error"
                    role="alert"
                    key={audit.auditId}
                  >
                    <p>
                      <strong>
                        Possible issues unavailable: {auditProfileLabel(audit)}{" "}
                        · {shortenId(audit.auditId)}
                      </strong>
                    </p>
                    <p>
                      {findings.results[index]?.isSuccess
                        ? "Showing the last loaded possible issues of this check; refreshing them failed."
                        : "This check is missing from the list below."}
                    </p>
                    <div className="issues-notice-actions">
                      <button
                        className="ui-btn"
                        data-size="xs"
                        type="button"
                        onClick={() => void findings.results[index]?.refetch()}
                      >
                        Retry
                      </button>
                    </div>
                  </div>
                );
              })}
              {visible.length === 0 && !loading && failed.length === 0 ? (
                <div className="issues-embedded-empty">
                  <EmptyState
                    title={
                      rows.length === 0
                        ? "No possible issues yet"
                        : "No matching possible issues"
                    }
                  >
                    <p>
                      {rows.length === 0
                        ? "Possible issues appear here as checks find them."
                        : "Try another search or clear the filters."}
                    </p>
                  </EmptyState>
                </div>
              ) : null}
              <AuditAnchor ready={!loading} />
              {visible.length === 0 ? null : (
                <div className="issues-embedded-list">
                  <ListSection>
                    {visible.map(({ audit, finding }) => (
                      <IssueRow
                        key={`${audit.auditId}:${finding.findingId}`}
                        id={`finding-${audit.auditId}-${finding.findingId}`}
                        finding={finding}
                        to={issueHref(finding, {
                          state,
                          project: projectId,
                          severity,
                        })}
                        context={
                          <ContextLink
                            returnLabel="Project possible issues"
                            to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/findings`}
                            title={`Check ${audit.auditId}`}
                          >
                            {/* Checks of one type differ only by their ID. */}
                            {auditProfileLabel(audit)} ·{" "}
                            {shortenId(audit.auditId)}
                          </ContextLink>
                        }
                      />
                    ))}
                  </ListSection>
                </div>
              )}
              <LoadMoreControl
                shown={rows.length}
                noun="possible issues"
                truncated={findings.truncated}
                loading={findings.isLoadingMore}
                error={moreFindings?.moreError ?? null}
                onLoadMore={findings.loadMore}
                label="Some checks have more possible issues — load more"
              />
              <LoadMoreControl
                shown={sources.length}
                noun="checks"
                truncated={audits.truncated}
                loading={audits.isLoadingMore}
                error={audits.moreError}
                onLoadMore={audits.loadMore}
                label="Load more checks"
              />
            </>
          )
        }
      </QueryView>
    </section>
  );
}
