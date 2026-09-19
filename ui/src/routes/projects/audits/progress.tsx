import { useQuery } from "@tanstack/react-query";
import { Link, useLocation } from "react-router";
import {
  auditNeedsPolling,
  getAuditWorkspace,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ErrorNotice, formatTimestamp } from "../../artifacts/common";

export function AuditProgress({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const location = useLocation();
  const summary = useQuery({
    queryKey: [...queryKeys.audits.detail(audit.auditId), "workspace"],
    queryFn: () => getAuditWorkspace(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 5_000 : false,
    refetchOnReconnect: true,
  });
  const value = summary.data;
  const root = `/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}`;
  const revision =
    value === undefined ? "" : `auditRevision=${value.auditRevision}`;
  const metrics = [
    {
      label: "Completed / total checks",
      count:
        value === undefined
          ? undefined
          : `${value.completedChecks} / ${value.totalChecks}`,
      to: `coverage?${revision}`,
    },
    {
      label: "Checks with issues",
      count: value?.issues,
      to: `coverage?result=issues&${revision}`,
    },
    {
      label: "Need follow-up",
      count: value?.gaps,
      to: `coverage?result=uncertain&${revision}`,
    },
    {
      label: "Not checked yet",
      count: value?.unchecked,
      to: `coverage?result=not-tested&${revision}`,
    },
    {
      label: "Unreviewed findings",
      count: value?.unreviewedFindings,
      to: `findings?verdict=unreviewed&${revision}`,
    },
    {
      label: "Pending decisions",
      count: value?.pendingReviews,
      to: `reviews?state=pending&${revision}`,
    },
  ];
  return (
    <section className="panel audit-section-panel" aria-label="Audit progress">
      <div className="section-heading">
        <div>
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
      {value !== undefined && value.totalChecks > 0 ? (
        <div
          className={`audit-coverage-outcome ${value.completedChecks < value.totalChecks || value.gaps > 0 || value.unchecked > 0 ? "is-partial" : ""}`}
        >
          <strong>
            {value.completedChecks < value.totalChecks ||
            value.gaps > 0 ||
            value.unchecked > 0
              ? "Partial coverage"
              : "Checks concluded"}
          </strong>
          <span>
            {value.completedChecks} of {value.totalChecks} checks concluded ·{" "}
            {Math.round((value.completedChecks / value.totalChecks) * 100)}%
          </span>
          <progress
            aria-label="Concluded checks"
            max={value.totalChecks}
            value={value.completedChecks}
          />
          <p>
            {value.gaps} need follow-up · {value.unchecked} not checked.
            Findings awaiting an analyst decision are candidates, not confirmed
            issues.
          </p>
        </div>
      ) : null}
      <div className="audit-progress-grid">
        {metrics.map((metric) => (
          <Link
            key={metric.label}
            className="audit-progress-stat"
            aria-label={`${metric.label}: ${metric.count ?? "unavailable"}`}
            to={`${root}/${metric.to}`}
            state={location.state}
          >
            <strong>{metric.count ?? "—"}</strong>
            <span>{metric.label} →</span>
          </Link>
        ))}
      </div>
      {value === undefined ? null : (
        <p className="muted-copy">
          Revision {value.auditRevision} · As of {formatTimestamp(value.asOf)} ·{" "}
          {value.roundId === undefined
            ? "No round yet"
            : `Round ${value.roundId}`}
          .
        </p>
      )}
      <p className="muted-copy">
        Execution: {value?.executionState ?? audit.state} ·{" "}
        {value?.outstandingRuns ?? audit.outstandingRunCount} outstanding Runs.
        Completed checks include concluded assessments and explicit exclusions.
        Finished execution does not imply complete coverage, accepted findings
        or an approved report.
      </p>
      {summary.isPending ? (
        <p role="status">Loading progress and decisions…</p>
      ) : null}
      {summary.error === null ? null : (
        <div>
          <ErrorNotice error={summary.error} />
          <p className="muted-copy">
            {value === undefined
              ? "Unavailable counts are shown as —."
              : "Counts show the last successful snapshot. Refresh before acting."}
          </p>
          <button
            className="secondary-button"
            type="button"
            disabled={summary.isFetching}
            onClick={() => void summary.refetch()}
          >
            Retry progress
          </button>
        </div>
      )}
    </section>
  );
}
