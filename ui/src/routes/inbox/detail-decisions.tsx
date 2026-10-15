import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";

import { getAuditReport, type AuditReport } from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { RecordedTime } from "../../app/recorded-time";
import {
  reportStatusLabel,
  reviewKindLabel,
  reviewStateLabel,
} from "../../app/vocabulary";
import { DetailPane, Facts, IdChip, TechnicalDetails } from "../../ui";
import {
  ActionDecision,
  FindingDecision,
  FindingSummary,
  ReportDecision,
} from "../decisions";
import { decisionOutcome } from "../decisions/model";
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
  itemLabel,
  pendingReviewsPath,
  projectPath,
  reportPath,
  reviewSubjectItem,
  reviewTitle,
  SECTIONS,
} from "./present";

/*
 * The Inbox details of items that wait for a decision (possible issues,
 * approvals, applicability, report acceptance) or are ready (reports).
 */

/** A possible issue: its summary, decided in place. */
export function IssueDetail({
  row,
  context,
}: {
  row: InboxRowOf<"issue">;
  context: DetailContext;
}) {
  const { project, audit, finding } = row.issue;
  const issues = context.data.model.decide.filter(
    (candidate) => candidate.type === "issue",
  );
  const position = issues.findIndex((candidate) => candidate.key === row.key);
  const { nextDecision, onSelect, onDecided } = context;
  const title = finding.firstProposal.document.title;
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
            <OpenLink to={issuePath(audit.auditId, finding.findingId)}>
              Open full review
            </OpenLink>
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
      footer={
        <FindingDecision
          auditId={audit.auditId}
          finding={finding}
          next={
            nextDecision === undefined
              ? undefined
              : {
                  label: "Next decision",
                  onNext: () => onSelect(nextDecision, { replace: true }),
                }
          }
          onDecided={(result) =>
            onDecided(row.key, decisionOutcome(result.decision).label, title)
          }
        />
      }
    >
      <Recorded text={context.recorded} />
      <p className="inbox-context">
        From{" "}
        <Link to={checkPath(project.projectId, audit.auditId)}>
          {checkTypeLabel(audit)}
        </Link>{" "}
        on <Link to={projectPath(project.projectId)}>{project.name}</Link>
      </p>
      <FindingSummary
        auditId={audit.auditId}
        finding={finding}
        variant="compact"
      />
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
        {report?.status === "proposed"
          ? "This report is proposed and not accepted yet. "
          : ""}
        A report is not a security or compliance certification.{" "}
        <ContextLink returnLabel="Inbox" to={reportPath(auditId)}>
          Open report
        </ContextLink>
      </p>
    </>
  );
}

/**
 * An approval, applicability or report acceptance request, decided in
 * place.
 */
export function ReviewDetail({
  row,
  context,
}: {
  row: InboxRowOf<"review">;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const { project, audit, review } = row.decision;
  const acceptance = review.kind === "report-acceptance";
  // The same read as ReportDecision, so both share one cached report.
  const report = useQuery({
    queryKey: queryKeys.audits.report(audit.auditId),
    queryFn: () => getAuditReport(api, audit.auditId),
    enabled: acceptance,
  });
  const item = reviewSubjectItem(row.decision, context.data.items.byCheck);
  const kindLabel = reviewKindLabel(review.kind);
  const title = reviewTitle(row.decision, item, kindLabel);
  const state = reviewStateLabel(review.state);
  const decided = context.onDecided;
  return (
    <DetailPane
      header={
        <DetailBar
          where={SECTIONS.decide.title}
          label={kindLabel}
          actions={
            acceptance ? (
              <OpenLink to={reportPath(audit.auditId)}>Open report</OpenLink>
            ) : (
              <OpenLink
                to={pendingReviewsPath(project.projectId, audit.auditId)}
              >
                Open in check
              </OpenLink>
            )
          }
          context={context}
          focus={context.focusKey === row.key}
        />
      }
      footer={
        acceptance ? (
          // A report that could not be read is explained here, with Try again.
          <ReportDecision
            auditId={audit.auditId}
            review={review}
            report={report.data}
            onDecided={(result) =>
              decided(row.key, decisionOutcome(result.decision).label, title)
            }
          />
        ) : (
          <ActionDecision
            auditId={audit.auditId}
            review={review}
            onDecided={(result) =>
              decided(row.key, decisionOutcome(result.decision).label, title)
            }
          />
        )
      }
    >
      <Recorded text={context.recorded} />
      <Head tone={state.tone} state={state.label} title={title} />
      <Facts
        className="inbox-facts"
        items={[
          {
            term: "Check",
            value: (
              <Link to={checkPath(project.projectId, audit.auditId)}>
                {checkTypeLabel(audit)}
              </Link>
            ),
          },
          {
            term: "Project",
            value: (
              <Link to={projectPath(project.projectId)}>{project.name}</Link>
            ),
          },
          {
            term: "Subject",
            value: acceptance
              ? "The check's proposed report"
              : item === undefined
                ? "A work item of this check"
                : itemLabel(item),
          },
          {
            term: "Requested",
            value: <RecordedTime value={review.createdAt} />,
          },
          {
            term: "Expires",
            value:
              review.expiresAt === undefined ? undefined : (
                <RecordedTime value={review.expiresAt} />
              ),
          },
        ]}
      />
      {!acceptance ? null : report.data !== undefined ? (
        <Block title="Proposed report">
          <ReportExcerpt auditId={audit.auditId} report={report.data} />
        </Block>
      ) : report.isPending ? (
        <Block title="Proposed report">
          <p className="ui-caption">Loading the report…</p>
        </Block>
      ) : // A failed read: the decision below says so once, with Try again.
      null}
      <TechnicalDetails description="Request and subject identifiers.">
        <Facts
          className="inbox-facts"
          size="sm"
          items={[
            {
              term: "Request ID",
              value: <IdChip value={review.requestId} label="request ID" />,
            },
            {
              term: "Subject ID",
              value: <IdChip value={review.subjectId} label="subject ID" />,
            },
            {
              term: "Check ID",
              value: <IdChip value={audit.auditId} label="check ID" />,
            },
            { term: "Subject revision", value: review.subjectRevision },
          ]}
        />
      </TechnicalDetails>
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
