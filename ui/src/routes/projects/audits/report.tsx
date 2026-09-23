import { useQuery } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import { Link, useSearchParams } from "react-router";

import {
  auditNeedsPolling,
  getAuditReport,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { StaleDataWarning } from "../../../app/query-view";
import { queryKeys } from "../../../api/query-keys";
import { StateBadge } from "../../runs/components";
import { useAuditProjectionRefresh } from "./projection-refresh";
import { AuditQueueError } from "./queue";
import { ActionReviewControls } from "./reviews";
import { ExactArtifactLink } from "./shared";

const MarkdownArtifactPreview = lazy(
  () => import("../../artifacts/previews/markdown"),
);

export function AuditReportView({
  audit,
  api,
}: {
  audit: Audit;
  api: ReturnType<typeof usePublicAPI>;
}) {
  const [params] = useSearchParams();
  const requestedReview = params.get("review");
  const queryKey = queryKeys.audits.report(audit.auditId);
  const report = useQuery({
    queryKey,
    queryFn: () => getAuditReport(api, audit.auditId),
    refetchInterval: auditNeedsPolling(audit.state) ? 1_000 : false,
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey);
  if (report.data === undefined) {
    if (report.error === null)
      return <p className="loading-copy">Loading report…</p>;
    return (
      <AuditQueueError
        error={report.error}
        onRefresh={() => void report.refetch()}
      />
    );
  }
  function downloadReport(
    name: string,
    mediaType: string,
    content: string,
  ): void {
    const url = URL.createObjectURL(new Blob([content], { type: mediaType }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  return (
    <section className="panel audit-section-panel audit-report-view">
      <div className="section-heading">
        <div>
          <h3>Audit report</h3>
        </div>
        <StateBadge state={report.data.status} />
      </div>
      {report.error === null ? null : (
        <StaleDataWarning
          error={report.error}
          onRetry={() => void report.refetch()}
          retryPending={report.isFetching}
        />
      )}
      <p className="muted-copy">
        Read the conclusion and its limitations. Full coverage and accepted
        findings are separate from execution status.
      </p>
      <Link
        to={`/projects/${encodeURIComponent(audit.projectId)}/audits/${encodeURIComponent(audit.auditId)}/coverage?result=uncertain`}
      >
        Review coverage gaps →
      </Link>
      {requestedReview !== null &&
      report.data.review?.requestId !== requestedReview ? (
        <div className="notice notice-error" role="status">
          The requested report review is unavailable or no longer current. This
          report cannot be used to decide that review.
        </div>
      ) : null}
      {report.data.status === "pending" ? (
        <p>
          The Audit has not reached report generation. Pending is not a
          successful assessment.
        </p>
      ) : report.data.status === "unavailable" ? (
        <div className="notice notice-error">
          <strong>No accepted report is available.</strong>
          <p>Inspect checks and collection dispositions for explicit gaps.</p>
        </div>
      ) : (
        <>
          {report.data.status === "proposed" ? (
            <div className="notice">
              <strong>This report is awaiting owner acceptance.</strong>
              <p>Review the contents below before making a decision.</p>
            </div>
          ) : null}
          {report.data.summary === undefined ? null : (
            <div className="audit-report-summary">
              <h4>Summary</h4>
              <div className="audit-report-markdown">
                <Suspense fallback={<p>Loading Markdown preview…</p>}>
                  <MarkdownArtifactPreview source={report.data.summary} />
                </Suspense>
              </div>
            </div>
          )}
          <div className="audit-artifact-list">
            {report.data.machineArtifact === undefined ? null : (
              <div className="audit-download-row">
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={report.data.machineArtifact}
                  label="Machine report"
                />
                {report.data.machine === undefined ? null : (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() =>
                      downloadReport(
                        `${audit.auditId}-report.json`,
                        "application/json",
                        JSON.stringify(report.data.machine, null, 2),
                      )
                    }
                  >
                    Download JSON
                  </button>
                )}
              </div>
            )}
            {report.data.summaryArtifact === undefined ? null : (
              <div className="audit-download-row">
                <ExactArtifactLink
                  projectId={audit.projectId}
                  artifact={report.data.summaryArtifact}
                  label="Summary report"
                />
                {report.data.summary === undefined ? null : (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() =>
                      downloadReport(
                        `${audit.auditId}-report.md`,
                        "text/markdown",
                        report.data.summary!,
                      )
                    }
                  >
                    Download summary
                  </button>
                )}
              </div>
            )}
          </div>
        </>
      )}
      {report.data.status === "proposed" &&
      report.data.review?.state === "pending" &&
      (requestedReview === null ||
        requestedReview === report.data.review.requestId) ? (
        <ActionReviewControls audit={audit} review={report.data.review} />
      ) : null}
    </section>
  );
}
