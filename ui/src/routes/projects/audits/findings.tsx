import { useQueries, useQuery } from "@tanstack/react-query";
import { Link, useParams, useSearchParams } from "react-router";

import { collectAuditPages } from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditFindings,
  listAuditReviews,
  listProjectAudits,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { getProject, PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../artifacts/common";
import { AuditFindingCard } from "./detail";
import { auditProfileLabel } from "./labels";
import { AuditAnchor, ProjectAuditNavigation } from "./shared";

import "./styles.css";

export function ProjectFindingsRoute() {
  const api = usePublicAPI();
  const { projectId = "" } = useParams();
  const [filters, setFilters] = useSearchParams();
  const validProject = PROJECT_ID_PATTERN.test(projectId);
  const project = useQuery({
    queryKey: queryKeys.projects.detail(projectId),
    queryFn: () => getProject(api, projectId),
    enabled: validProject,
  });
  const audits = useQuery({
    queryKey: queryKeys.projects.audits.inventory(projectId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listProjectAudits(api, {
          projectId,
          ...(cursor === undefined ? {} : { cursor }),
        }),
      ),
    enabled: validProject && project.data?.kind === "project",
    refetchInterval: 5_000,
    refetchOnReconnect: true,
  });
  const sources = audits.data ?? [];
  const findings = useQueries({
    queries: sources.map((audit) => ({
      queryKey: queryKeys.audits.allFindings(audit.auditId),
      queryFn: () =>
        collectAuditPages((cursor) =>
          listAuditFindings(
            api,
            audit.auditId,
            cursor === undefined ? {} : { cursor },
          ),
        ),
      refetchInterval:
        auditNeedsPolling(audit.state) || audit.outstandingRunCount > 0
          ? 5_000
          : (false as const),
      refetchOnReconnect: true,
    })),
  });
  const reviews = useQueries({
    queries: sources.map((audit) => ({
      queryKey: queryKeys.audits.allReviews(audit.auditId),
      queryFn: () =>
        collectAuditPages((cursor) =>
          listAuditReviews(
            api,
            audit.auditId,
            cursor === undefined ? {} : { cursor },
          ),
        ),
      refetchInterval:
        auditNeedsPolling(audit.state) || audit.outstandingRunCount > 0
          ? 5_000
          : (false as const),
      refetchOnReconnect: true,
    })),
  });
  const query = (filters.get("q") ?? "").trim().toLocaleLowerCase();
  const selectedAudit = filters.get("audit") ?? "";
  const severity = filters.get("severity") ?? "";
  const rows = sources.flatMap((audit, index) =>
    (findings[index]?.data ?? []).map((finding) => ({ audit, finding, index })),
  );
  const visible = rows
    .filter(({ audit, finding }) => {
      const document = finding.firstProposal.document;
      return (
        (selectedAudit === "" || audit.auditId === selectedAudit) &&
        (severity === "" ||
          (finding.analystSeverity ?? document.severity_suggestion) ===
            severity) &&
        (query === "" ||
          [
            document.title,
            document.description,
            document.subject.key,
            finding.findingId,
            auditProfileLabel(audit),
          ]
            .join(" ")
            .toLocaleLowerCase()
            .includes(query))
      );
    })
    .sort(
      (a, b) =>
        b.finding.createdAt.localeCompare(a.finding.createdAt) ||
        a.finding.findingId.localeCompare(b.finding.findingId),
    );
  const loading = findings.some((result) => result.isPending);
  const refreshing =
    audits.isFetching ||
    findings.some((result) => result.isFetching) ||
    reviews.some((result) => result.isFetching);
  const failed = sources.filter((_, index) => findings[index]?.isError);

  function setFilter(name: string, value: string) {
    setFilters(
      (current) => {
        const next = new URLSearchParams(current);
        if (value === "") next.delete(name);
        else next.set(name, value);
        return next;
      },
      { replace: true },
    );
  }

  if (!validProject)
    return (
      <section className="route-page">
        <ErrorNotice error={new Error("Project Findings route is invalid")} />
        <Link to="/projects">Return to Projects</Link>
      </section>
    );

  return (
    <section className="route-page audit-page project-findings-page">
      <header className="route-header-row">
        <div>
          <Link
            className="back-link"
            to={`/projects/${encodeURIComponent(projectId)}`}
          >
            ← {project.data?.name ?? "Project"}
          </Link>
          <p className="eyebrow">Project findings</p>
          <h2>Findings</h2>
          <p className="lede">
            Findings from every audit in this project, with their source,
            evidence and review status.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={refreshing}
          onClick={() =>
            void Promise.all([
              audits.refetch(),
              ...findings.map((result) => result.refetch()),
              ...reviews.map((result) => result.refetch()),
            ])
          }
        >
          {refreshing ? "Refreshing…" : "Refresh"}
        </button>
      </header>
      <ProjectAuditNavigation projectId={projectId} current="findings" />
      {project.error !== null ? (
        <ErrorNotice error={project.error} />
      ) : project.isPending ? (
        <p className="loading-copy">Loading Project…</p>
      ) : project.data.kind !== "project" ? (
        <ErrorNotice
          error={new Error("Findings are available for Projects.")}
        />
      ) : audits.error !== null ? (
        <ErrorNotice error={audits.error} />
      ) : audits.isPending ? (
        <p className="loading-copy">Loading Audits…</p>
      ) : (
        <>
          <section
            className="panel audit-findings-toolbar"
            aria-label="Finding filters"
          >
            <label>
              Search findings
              <input
                type="search"
                value={filters.get("q") ?? ""}
                onChange={(event) => setFilter("q", event.target.value)}
                placeholder="Title, description or affected component"
              />
            </label>
            <label>
              Audit
              <select
                value={selectedAudit}
                onChange={(event) => setFilter("audit", event.target.value)}
              >
                <option value="">All audits ({sources.length})</option>
                {sources.map((audit) => (
                  <option key={audit.auditId} value={audit.auditId}>
                    {auditProfileLabel(audit)} · {audit.auditId.slice(-8)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Severity
              <select
                value={severity}
                onChange={(event) => setFilter("severity", event.target.value)}
              >
                <option value="">All severities</option>
                {["critical", "high", "medium", "low", "informational"].map(
                  (value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ),
                )}
              </select>
            </label>
          </section>
          <div className="section-heading audit-findings-count" role="status">
            <span>
              {visible.length} of {rows.length} findings
              {loading || failed.length > 0 ? " loaded" : ""} · {sources.length}{" "}
              audits
            </span>
            {selectedAudit || severity || query ? (
              <button
                className="secondary-button"
                type="button"
                onClick={() => setFilters({}, { replace: true })}
              >
                Clear filters
              </button>
            ) : null}
          </div>
          {loading ? (
            <p className="loading-copy">Loading findings from all audits…</p>
          ) : null}
          {failed.map((audit) => {
            const index = sources.indexOf(audit);
            return (
              <div
                className="notice notice-error"
                role="alert"
                key={audit.auditId}
              >
                <strong>
                  Findings unavailable: {auditProfileLabel(audit)} ·{" "}
                  {audit.auditId.slice(-8)}
                </strong>
                <p>
                  {findings[index]?.data === undefined
                    ? "This audit is missing from the results shown below."
                    : "Showing the last loaded findings for this audit. Refresh failed."}
                </p>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => void findings[index]?.refetch()}
                >
                  Retry findings
                </button>
              </div>
            );
          })}
          {visible.length === 0 && !loading && failed.length === 0 ? (
            <div className="empty-state panel">
              <h3>
                {rows.length === 0 ? "No findings yet" : "No matching findings"}
              </h3>
              <p>
                {rows.length === 0
                  ? "Findings will appear here as audit checks publish them."
                  : "Try another search or clear the filters."}
              </p>
            </div>
          ) : null}
          <div className="audit-finding-list">
            <AuditAnchor ready={!loading} />
            {visible.map(({ audit, finding, index }) => {
              const reviewQuery = reviews[index]!;
              const pendingReview = reviewQuery.data?.find(
                (review) =>
                  review.state === "pending" &&
                  review.findingId === finding.findingId,
              );
              return (
                <AuditFindingCard
                  key={`${audit.auditId}:${finding.findingId}`}
                  audit={audit}
                  finding={finding}
                  findings={findings[index]!.data ?? []}
                  showAudit
                  {...(pendingReview === undefined ? {} : { pendingReview })}
                  reviewLoading={reviewQuery.isPending}
                  reviewError={reviewQuery.error}
                  onRetryReview={() => void reviewQuery.refetch()}
                />
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
