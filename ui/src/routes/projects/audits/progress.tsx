import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router";
import { collectAuditPages } from "../../../api/audit-collections";
import {
  auditNeedsPolling,
  listAuditReviews,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice } from "../../artifacts/common";
import { ASSESSMENTS } from "./assessments";
import { useAuditCoverage } from "./coverage-data";

export function AuditProgress({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const location = useLocation();
  const coverage = useAuditCoverage(audit);
  const reviews = useQuery({
    queryKey: queryKeys.audits.allReviews(audit.auditId),
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditReviews(
          api,
          audit.auditId,
          cursor === undefined ? {} : { cursor },
        ),
      ),
    refetchInterval: auditNeedsPolling(audit.state) ? 5_000 : false,
    refetchOnReconnect: true,
  });
  const root = `/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}`;
  const pending = reviews.data?.filter((review) => review.state === "pending");
  const metrics = [
    {
      label: "All checks",
      count: coverage.data?.length,
      to: `${root}/coverage`,
    },
    ...(
      [
        ["issues", "Issues found"],
        ["uncertain", "Need follow-up"],
        ["not-tested", "Not checked yet"],
      ] as const
    ).map(([group, label]) => ({
      label,
      count: coverage.data?.filter(
        (row) => ASSESSMENTS[row.coverage.status].group === group,
      ).length,
      to: `${root}/coverage?result=${group}`,
    })),
    {
      label: "Pending decisions",
      count: pending?.length,
      to: `${root}/reviews?state=pending${pending?.length === 1 ? `#review-${encodeURIComponent(pending[0]!.requestId)}` : ""}`,
    },
  ];
  return (
    <section className="panel audit-section-panel" aria-label="Audit progress">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Current round</p>
          <h3>Progress and decisions</h3>
        </div>
        <Link
          className="audit-open-link"
          to={`${root}/report`}
          state={location.state}
        >
          Open report →
        </Link>
      </div>
      <div className="audit-progress-grid">
        {metrics.map((metric) => (
          <Link
            key={metric.label}
            className="audit-progress-stat"
            aria-label={`${metric.label}: ${metric.count ?? "unavailable"}`}
            to={metric.to}
            state={location.state}
          >
            <strong>{metric.count ?? "—"}</strong>
            <span>{metric.label} →</span>
          </Link>
        ))}
      </div>
      <p className="muted-copy">
        Execution: {audit.state} · {audit.outstandingRunCount} outstanding Runs.
        Finished execution does not imply complete coverage or an accepted
        report.
      </p>
      {coverage.isPending || reviews.isPending ? (
        <p role="status">Loading progress and decisions…</p>
      ) : null}
      {[coverage, reviews].map((query, index) =>
        query.error === null ? null : (
          <div key={index}>
            <ErrorNotice error={query.error} />
            <p className="muted-copy">
              {query.data === undefined
                ? "Unavailable counts are shown as —."
                : "Counts show the last successful snapshot."}
            </p>
            <button
              className="secondary-button"
              type="button"
              disabled={query.isFetching}
              onClick={() => void query.refetch()}
            >
              Retry {index === 0 ? "progress" : "decisions"}
            </button>
          </div>
        ),
      )}
    </section>
  );
}
