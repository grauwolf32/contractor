import { useId } from "react";
import { useSearchParams } from "react-router";

import type { Audit } from "../../../api/audits";
import type { PublicAPI } from "../../../api/client";
import { useAuditReport, useReportAcceptance } from "../../reports/report-data";
import {
  ReportAcceptanceDecision,
  ReportContent,
  ReportStatusChip,
} from "../../reports/report-view";

function CheckReport({ audit, api }: { audit: Audit; api: PublicAPI }) {
  const [params] = useSearchParams();
  const query = useAuditReport(audit, api);
  // `?review=<requestId>` comes from review links: only that request is
  // offered, and a report that does not carry it says so.
  const acceptance = useReportAcceptance(
    api,
    audit.auditId,
    query.data,
    params.get("review"),
  );
  const headingId = useId();
  const decisionHeadingId = useId();
  return (
    <section className="reports-check" aria-labelledby={headingId}>
      <header className="reports-check-head">
        <h2 id={headingId} className="reports-check-title">
          Report
        </h2>
        {query.data === undefined ? null : (
          <ReportStatusChip status={query.data.status} />
        )}
      </header>
      <ReportContent audit={audit} query={query} acceptance={acceptance} />
      {acceptance.review === undefined ? null : (
        <section
          className="reports-check-decision"
          aria-labelledby={decisionHeadingId}
        >
          <h3 id={decisionHeadingId} className="reports-heading">
            Report acceptance
          </h3>
          <div className="decisions-inline">
            <ReportAcceptanceDecision
              auditId={audit.auditId}
              acceptance={acceptance}
            />
          </div>
        </section>
      )}
    </section>
  );
}

/**
 * The check page's Report section: the same report content as /reports, with
 * the acceptance decision inline.
 */
export function AuditReportView({
  audit,
  api,
}: {
  audit: Audit;
  api: PublicAPI;
}) {
  // A kept acceptance request never crosses checks.
  return <CheckReport key={audit.auditId} audit={audit} api={api} />;
}
