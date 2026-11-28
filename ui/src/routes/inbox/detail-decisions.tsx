import { Link, useSearchParams } from "react-router";

import type { Audit } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { Project } from "../../api/projects";
import { RecordedTime } from "../../app/recorded-time";
import { verdictLabel } from "../../app/vocabulary";
import { DetailHeader, DetailPane } from "../../ui";
import { IssueDetail as FullIssueDetail } from "../issues/issue-detail";
import { checkPath, sectionPath } from "../projects/audits/check-links";
import { useAuditReport } from "../reports/report-data";
import {
  ReportContent,
  ReportStatusChip,
} from "../../widgets/audit-report/view";
import {
  DetailBar,
  OpenLink,
  Recorded,
  type DetailContext,
} from "./detail-parts";
import type { InboxRowOf } from "./model";
import { checkTypeLabel, issuePath, projectPath, reportPath } from "./present";

/** Inbox supplies queue context; Issues owns the complete reader and annotation. */
export function IssueDetail({
  row,
  context,
}: {
  row: InboxRowOf<"issue">;
  context: DetailContext;
}) {
  const [params] = useSearchParams();
  const selected = row.issue.finding;
  const issues = context.data.model.decide.filter(
    (candidate) => candidate.type === "issue",
  );
  const position = issues.findIndex((candidate) => candidate.key === row.key);
  const tab = params.get("tab");
  const search = tab === "evidence" || tab === "history" ? `?tab=${tab}` : "";
  return (
    <div className="issues-workspace inbox-full-issue">
      <Recorded text={context.recorded} />
      <DetailBar
        where="Decide"
        label={
          position < 0
            ? "Possible issue"
            : `Possible issue ${position + 1} of ${issues.length}`
        }
        actions={
          <>
            <OpenLink to={projectPath(selected.projectId) + "/findings"}>
              {row.issue.project.name}
            </OpenLink>
            <OpenLink
              to={issuePath(selected.projectId, selected.findingId) + search}
            >
              Open in Issues
            </OpenLink>
          </>
        }
        context={context}
        focus={context.focusKey === row.key}
      />
      <FullIssueDetail
        key={row.key}
        projectId={selected.projectId}
        findingId={selected.findingId}
        listed={row.issue}
        projects={[row.issue.project]}
        position={
          position < 0 ? undefined : { index: position, count: issues.length }
        }
        onDecided={(result) =>
          context.onDecided(
            row.key,
            verdictLabel(result.finding.analystVerdict).label,
            selected.title,
          )
        }
      />
    </div>
  );
}

/** Reports owns the read/status/Markdown/downloads, including unavailable reads. */
export function ReportDetail({
  audit,
  project,
  row,
  context,
}: {
  audit: Audit;
  project: Project;
  row?: InboxRowOf<"report"> | undefined;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const query = useAuditReport(audit, api);
  return (
    <DetailPane
      header={
        <>
          <DetailBar
            where="Ready"
            label="Report"
            actions={
              <OpenLink to={reportPath(audit.auditId)}>Open report</OpenLink>
            }
            context={context}
            focus={row !== undefined && context.focusKey === row.key}
          />
          <DetailHeader
            title={`${project.name} report`}
            status={
              query.data === undefined ? undefined : (
                <ReportStatusChip status={query.data.status} />
              )
            }
            meta={
              <>
                <Link to={checkPath(project.projectId, audit.auditId)}>
                  {checkTypeLabel(audit)}
                </Link>
                <span>
                  Updated <RecordedTime value={audit.updatedAt} />
                </span>
              </>
            }
          />
        </>
      }
    >
      <Recorded text={context.recorded} />
      <ReportContent
        audit={audit}
        query={query}
        links={{
          taskResults: checkPath(audit.projectId, audit.auditId),
          coverageGaps: `${sectionPath(audit.projectId, audit.auditId, "work")}?work=gap`,
        }}
      />
    </DetailPane>
  );
}
