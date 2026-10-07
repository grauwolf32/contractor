import type { UseQueryResult } from "@tanstack/react-query";
import { useId, useMemo, useState, type ReactNode } from "react";
import { Link } from "react-router";

import type {
  Audit,
  AuditFinding,
  AuditReport,
  AuditReviewPage,
  AuditWorkspace,
} from "../../../api/audits";
import { ContextLink } from "../../../app/context-navigation";
import { RecordedTime } from "../../../app/recorded-time";
import {
  findingStateLabel,
  reportStatusLabel,
  reviewActionLabel,
  reviewKindLabel,
  type ItemKind,
} from "../../../app/vocabulary";
import { ActivityLog, StatusChip, StatusGlyph } from "../../../ui";
import { ActionDecision } from "../../decisions";
import {
  CHECK_ACTIVITY_LIMIT,
  checkActivity,
  nowSentence,
} from "./activity-model";
import type { CheckLinks } from "./check-links";
import { REPORT_STATES } from "./check-data";
import type { CheckEntry } from "./check-model";
import { CheckTechnicalDetails } from "./overview";

/** Decisions shown in place before "See all decisions". */
const INLINE_DECISIONS = 3;
/** Possible issues listed on the activity view. */
const LISTED_ISSUES = 5;

function plural(count: number, singular: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? singular : many}`;
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  const heading = useId();
  return (
    <section className="checks-block" aria-labelledby={heading}>
      <h3 id={heading}>{title}</h3>
      {children}
    </section>
  );
}

function issuePath(auditId: string, findingId: string): string {
  return `/issues/${encodeURIComponent(auditId)}/${encodeURIComponent(findingId)}`;
}

function newestFirst(left: AuditFinding, right: AuditFinding): number {
  return Date.parse(right.createdAt) - Date.parse(left.createdAt);
}

function Decisions({
  audit,
  waiting,
  subjects,
  links,
}: {
  audit: Audit;
  waiting: UseQueryResult<AuditReviewPage>;
  subjects: ReadonlyMap<string, string>;
  links: CheckLinks;
}) {
  // The decision leaves this list once recorded; say so here.
  const [recorded, setRecorded] = useState("");
  const requests = waiting.data?.items ?? [];
  const total = waiting.data?.total ?? requests.length;
  if (waiting.isPending && audit.state !== "draft")
    return (
      <Block title="Decisions waiting">
        <p className="checks-quiet" role="status">
          Loading decisions…
        </p>
      </Block>
    );
  return (
    <Block title="Decisions waiting">
      <p className="checks-quiet" role="status">
        {recorded}
      </p>
      {waiting.error === null ? null : (
        <div className="checks-notice" data-tone="warning" role="status">
          <p>The decisions waiting for you could not be loaded.</p>
          <button
            className="ui-btn"
            data-size="xs"
            type="button"
            disabled={waiting.isFetching}
            onClick={() => void waiting.refetch()}
          >
            Try again
          </button>
        </div>
      )}
      {requests.length === 0 ? (
        waiting.error === null ? (
          <p className="checks-quiet">Nothing is waiting for you.</p>
        ) : null
      ) : (
        <ul className="checks-waiting">
          {requests.slice(0, INLINE_DECISIONS).map((review) => (
            <li key={review.requestId} className="checks-waiting-item">
              <p className="checks-waiting-title">
                <StatusGlyph tone="review" />
                <strong>{reviewKindLabel(review.kind)}</strong>
                <span>
                  {subjects.get(review.subjectId) ??
                    (review.subjectKind === "finding"
                      ? "Possible issue"
                      : review.subjectKind === "audit-report"
                        ? "Check report"
                        : "")}
                </span>
                <span className="checks-quiet">
                  Requested <RecordedTime value={review.createdAt} />
                </span>
              </p>
              {review.subjectKind === "audit-item-action" ? (
                <div className="decisions-inline">
                  <ActionDecision
                    auditId={audit.auditId}
                    review={review}
                    onDecided={(result) =>
                      setRecorded(
                        result.decision.action === undefined
                          ? "Decision recorded."
                          : `Decision recorded: ${reviewActionLabel(result.decision.action).label}.`,
                      )
                    }
                  />
                </div>
              ) : review.subjectKind === "finding" ? (
                <Link
                  to={issuePath(
                    audit.auditId,
                    review.findingId ?? review.subjectId,
                  )}
                >
                  Review the possible issue →
                </Link>
              ) : (
                <ContextLink
                  returnLabel="Check activity"
                  to={links.deep("report", { review: review.requestId })}
                >
                  Review the report →
                </ContextLink>
              )}
            </li>
          ))}
        </ul>
      )}
      {total > Math.min(requests.length, INLINE_DECISIONS) ? (
        <Link to={links.deep("reviews", { state: "pending" })}>
          See all {plural(total, "decision", "decisions")} waiting →
        </Link>
      ) : null}
    </Block>
  );
}

function PossibleIssues({
  audit,
  findings,
  workspace,
  links,
}: {
  audit: Audit;
  findings: readonly AuditFinding[];
  workspace: AuditWorkspace | undefined;
  links: CheckLinks;
}) {
  const latest = useMemo(
    () => [...findings].sort(newestFirst).slice(0, LISTED_ISSUES),
    [findings],
  );
  const total = workspace?.findings ?? findings.length;
  const unreviewed = workspace?.unreviewedFindings;
  return (
    <Block title="Possible issues">
      <p className="checks-quiet">
        {total === 0
          ? "No possible issues so far."
          : `${plural(total, "possible issue", "possible issues")} in this check${unreviewed === undefined ? "" : `, ${unreviewed.toLocaleString("en-US")} not reviewed yet`}. Possible issues stay separate from confirmed issues.`}
      </p>
      {latest.length === 0 ? null : (
        <ul className="checks-issues">
          {latest.map((finding) => {
            const state = findingStateLabel(finding.state);
            return (
              <li key={finding.findingId} className="checks-issue">
                <StatusGlyph tone={state.tone} />
                <div className="checks-issue-main">
                  <Link to={issuePath(audit.auditId, finding.findingId)}>
                    {finding.firstProposal.document.title}
                  </Link>
                  <p className="checks-issue-meta">
                    <span>
                      Found <RecordedTime value={finding.createdAt} />
                    </span>
                  </p>
                </div>
                <StatusChip tone={state.tone} size="sm">
                  {state.label}
                </StatusChip>
              </li>
            );
          })}
        </ul>
      )}
      {total === 0 ? null : (
        <p className="checks-links">
          <Link to={links.section("findings")}>
            All possible issues of this check →
          </Link>
        </p>
      )}
    </Block>
  );
}

function Report({
  audit,
  report,
  links,
}: {
  audit: Audit;
  report: UseQueryResult<AuditReport>;
  links: CheckLinks;
}) {
  if (!REPORT_STATES.includes(audit.state))
    return (
      <Block title="Report">
        <p className="checks-quiet">
          The report is written when the check finishes.
        </p>
      </Block>
    );
  const status =
    report.data === undefined
      ? undefined
      : reportStatusLabel(report.data.status);
  return (
    <Block title="Report">
      <p className="checks-report-line">
        {status === undefined ? (
          <span className="checks-quiet">
            {report.error === null
              ? "Loading the report status…"
              : "The report status could not be loaded."}
          </span>
        ) : (
          <StatusChip tone={status.tone} size="sm">
            {status.label}
          </StatusChip>
        )}
        <Link to={links.section("report")}>Open the report →</Link>
      </p>
    </Block>
  );
}

/**
 * The check as a whole: what it is doing now, the decisions waiting for the
 * user (active test approvals and applicability decided in place), its
 * possible issues, its report, everything that happened, and the technical
 * details.
 */
export function CheckActivity({
  audit,
  kind,
  entries,
  findings,
  workspace,
  waiting,
  report,
  subjects,
  links,
  technicalOpen = false,
}: {
  audit: Audit;
  kind: ItemKind;
  entries: readonly CheckEntry[];
  findings: readonly AuditFinding[];
  workspace: AuditWorkspace | undefined;
  waiting: UseQueryResult<AuditReviewPage>;
  report: UseQueryResult<AuditReport>;
  subjects: ReadonlyMap<string, string>;
  links: CheckLinks;
  /** Open the technical details (a "#technical-details" link). */
  technicalOpen?: boolean;
}) {
  const now = nowSentence(audit, entries, kind);
  const log = useMemo(
    () =>
      checkActivity({
        audit,
        entries,
        findings,
        waiting: waiting.data?.items ?? [],
        now,
      }),
    [audit, entries, findings, now, waiting.data],
  );
  return (
    <div className="checks-activity">
      <div className="checks-section-heading">
        <h2>All activity</h2>
        <p className="checks-quiet">
          What the whole check is doing, in plain words.
        </p>
      </div>
      <Decisions
        audit={audit}
        waiting={waiting}
        subjects={subjects}
        links={links}
      />
      <div className="checks-activity-grid">
        <PossibleIssues
          audit={audit}
          findings={findings}
          workspace={workspace}
          links={links}
        />
        <Report audit={audit} report={report} links={links} />
      </div>
      <Block title="Activity on this check">
        <ActivityLog
          aria-label="Activity on this check"
          entries={log.entries}
        />
        {log.total > CHECK_ACTIVITY_LIMIT ? (
          <p className="checks-quiet">
            Showing the latest {CHECK_ACTIVITY_LIMIT} of{" "}
            {log.total.toLocaleString("en-US")} events.
          </p>
        ) : null}
      </Block>
      <CheckTechnicalDetails
        audit={audit}
        workspace={workspace}
        open={technicalOpen}
      />
    </div>
  );
}
