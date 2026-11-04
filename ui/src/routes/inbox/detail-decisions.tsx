import { pollInterval } from "../../app/polling";
import { QueryView } from "../../app/query-view";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";

import { getProjectFinding, type AuditReport } from "../../api/audits";
import { usePublicAPI } from "../../api/context";

import { ContextLink } from "../../app/context-navigation";
import { RecordedTime } from "../../app/recorded-time";
import { verdictLabel, reportStatusLabel } from "../../app/vocabulary";
import { DetailPane, IdChip } from "../../ui";
import { FindingDecision, FindingSummary } from "../decisions";

import { firstParagraph, plainText } from "../decisions/text";
import { checkPath } from "../projects/audits/check-links";
import {
  Block,
  DetailBar,
  Head,
  OpenLink,
  Recorded,
  type DetailContext,
} from "./detail-parts";
import type { InboxRowOf } from "./model";
import {
  checkTypeLabel,
  excerpt,
  issuePath,
  projectPath,
  reportPath,
  SECTIONS,
} from "./present";

/*
 * Inbox details for optional Project Finding annotations and ready reports.
 */

/** A possible issue: its summary, decided in place. */
export function IssueDetail({
  row,
  context,
}: {
  row: InboxRowOf<"issue">;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const selected = row.issue.finding;
  const finding = useQuery({
    queryKey: ["project-findings", selected.projectId, selected.findingId],
    queryFn: () =>
      getProjectFinding(api, selected.projectId, selected.findingId),
    retry: false,
    refetchInterval: pollInterval({ interval: 20_000 }),
  });
  const issues = context.data.model.decide.filter(
    (candidate) => candidate.type === "issue",
  );
  const position = issues.findIndex((candidate) => candidate.key === row.key);
  const { nextDecision, onSelect, onDecided } = context;
  return (
    <DetailPane
      header={
        <DetailBar
          where={SECTIONS.decide.title}
          label={
            position < 0
              ? "Possible issue"
              : `Possible issue ${position + 1} of ${issues.length}`
          }
          actions={
            <OpenLink to={issuePath(selected.projectId, selected.findingId)}>
              Open possible issue
            </OpenLink>
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
      footer={
        finding.data === undefined ? undefined : (
          <FindingDecision
            finding={finding.data}
            next={
              nextDecision === undefined
                ? undefined
                : {
                    label: "Next decision",
                    onNext: () => onSelect(nextDecision, { replace: true }),
                  }
            }
            onDecided={(result) =>
              onDecided(
                row.key,
                verdictLabel(result.finding.analystVerdict).label,
                selected.title,
              )
            }
          />
        )
      }
    >
      <Recorded text={context.recorded} />
      <p className="inbox-context">
        In{" "}
        <Link to={projectPath(selected.projectId)}>
          {row.issue.project.name}
        </Link>
      </p>
      <QueryView
        query={finding}
        loading={<p role="status">Loading possible issue…</p>}
        onRetry={() => void finding.refetch()}
      >
        {(value) => <FindingSummary finding={value} variant="compact" />}
      </QueryView>
    </DetailPane>
  );
}

/** A ready report: its summary and the check it belongs to. */
export function ReportDetail({
  row,
  context,
}: {
  row: InboxRowOf<"report">;
  context: DetailContext;
}) {
  const { project, audit, report } = row.report;
  const status = reportStatusLabel(report.status);
  return (
    <DetailPane
      header={
        <DetailBar
          where={SECTIONS.ready.title}
          label="Report"
          actions={
            <OpenLink to={reportPath(audit.auditId)}>Open report</OpenLink>
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
    >
      <Recorded text={context.recorded} />
      <Head
        tone={status.tone}
        state={status.label}
        title={`${project.name} report`}
        meta={
          <>
            <Link to={checkPath(project.projectId, audit.auditId)}>
              {checkTypeLabel(audit)}
            </Link>
            <IdChip value={audit.auditId} label="check ID" />
            <span>
              Finished{" "}
              <RecordedTime value={audit.finishedAt ?? audit.updatedAt} />
            </span>
          </>
        }
      />
      <Block title="Summary">
        <ReportExcerpt auditId={audit.auditId} report={report} />
      </Block>
    </DetailPane>
  );
}

function ReportExcerpt({
  auditId,
  report,
}: {
  auditId: string;
  report: AuditReport | undefined;
}) {
  const summary =
    report?.summary === undefined
      ? ""
      : excerpt(plainText(firstParagraph(report.summary)), 600);
  return (
    <>
      {summary === "" ? null : <p className="inbox-excerpt">{summary}</p>}
      <p className="ui-caption">
        A report is not a security or compliance certification.{" "}
        <ContextLink returnLabel="Inbox" to={reportPath(auditId)}>
          Open report
        </ContextLink>
      </p>
    </>
  );
}
