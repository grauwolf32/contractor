import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router";

import { AUDIT_ID_PATTERN, type Audit } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { PublicAPIError } from "../../api/error";
import { getProject } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ErrorNotice } from "../../app/error-notice";
import { RecordedTime } from "../../app/recorded-time";
import { DetailHeader, DetailPane, EmptyState, IdChip } from "../../ui";
import { auditProfileLabel } from "../projects/audits/labels";
import { checkPath } from "./filters";
import { useAuditReport, useReportAcceptance } from "./report-data";
import {
  ReportAcceptanceDecision,
  ReportContent,
  ReportStatusChip,
} from "./report-view";

import "./reports.css";

/** The detail pane when no report is selected. */
export function NoReportSelected() {
  return (
    <DetailPane>
      <EmptyState title="Choose a report">
        <p>
          Pick a report to read its summary, download it, or decide whether to
          accept it.
        </p>
      </EmptyState>
    </DetailPane>
  );
}

function LoadedReport({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const [params] = useSearchParams();
  const query = useAuditReport(audit, api);
  const acceptance = useReportAcceptance(
    api,
    audit.auditId,
    query.data,
    params.get("review"),
  );
  const project = useQuery({
    queryKey: queryKeys.projects.detail(audit.projectId),
    queryFn: () => getProject(api, audit.projectId),
  });
  return (
    <DetailPane
      header={
        <DetailHeader
          title={auditProfileLabel(audit)}
          status={
            query.data === undefined ? undefined : (
              <ReportStatusChip status={query.data.status} />
            )
          }
          meta={
            <>
              <Link to={`/projects/${encodeURIComponent(audit.projectId)}`}>
                {project.data?.name ?? "Project"}
              </Link>
              <span className="reports-meta-id">
                Check ID <IdChip value={audit.auditId} label="check ID" />
              </span>
              <span>
                Updated <RecordedTime value={audit.updatedAt} />
              </span>
            </>
          }
          actions={
            <Link className="ui-btn" data-size="sm" to={checkPath(audit)}>
              Open check
            </Link>
          }
        />
      }
      footer={
        acceptance.review === undefined ? undefined : (
          <div className="reports-decision-footer">
            <ReportAcceptanceDecision
              auditId={audit.auditId}
              acceptance={acceptance}
            />
          </div>
        )
      }
    >
      <ReportContent audit={audit} query={query} acceptance={acceptance} />
    </DetailPane>
  );
}

/**
 * /reports/:auditId: the check's report with its header, downloads and,
 * while the report waits for acceptance, the decision. `check` is the
 * check's read (useReportCheck), shared with the page title.
 */
export function ReportDetail({
  auditId,
  check,
}: {
  auditId: string;
  check: UseQueryResult<Audit>;
}) {
  if (!AUDIT_ID_PATTERN.test(auditId))
    return (
      <DetailPane>
        <EmptyState title="This report link is not valid">
          <p>Choose a report from the list.</p>
        </EmptyState>
      </DetailPane>
    );
  if (check.data === undefined) {
    if (check.error === null)
      return (
        <DetailPane>
          <p className="reports-quiet" role="status">
            Loading the report…
          </p>
        </DetailPane>
      );
    if (check.error instanceof PublicAPIError && check.error.status === 404)
      return (
        <DetailPane>
          <EmptyState title="This check is not available">
            <p>It may have been deleted, or it is not yours to read.</p>
          </EmptyState>
        </DetailPane>
      );
    return (
      <DetailPane>
        <ErrorNotice
          error={check.error}
          context="Could not load the check"
          onRetry={() => void check.refetch()}
          retryPending={check.isFetching}
        />
      </DetailPane>
    );
  }
  return <LoadedReport audit={check.data} />;
}
