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
import { findingsReadable, REPORT_STATES } from "./check-data";
import { issueSummary, splitIssues, type CheckEntry } from "./check-model";
import type { AuditCollectionQuery } from "./collections";
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

/** "This check has 1 issue and 2 possible issues. 1 was marked …" */
function issuesSentence(
  findings: readonly AuditFinding[],
  partial: boolean,
  setAside: number,
): string {
  const summary = issueSummary(findings).join(" and ");
  const read = findings.length.toLocaleString("en-US");
  const counted =
    summary === ""
      ? partial
        ? `No issues or possible issues among the first ${read} read.`
        : "This check has no issues or possible issues."
      : `This check has ${partial ? "at least " : ""}${summary}.`;
  return setAside === 0
    ? counted
    : `${counted} ${setAside.toLocaleString("en-US")} ${setAside === 1 ? "was" : "were"} marked not an issue or duplicate.`;
}

/**
 * The check's issues and possible issues, kept apart by their state: the
 * counts of each, and the newest of them. Findings marked not an issue or
 * duplicate are counted, not listed.
 */
function Issues({
  audit,
  findings,
  links,
}: {
  audit: Audit;
  findings: AuditCollectionQuery<AuditFinding>;
  links: CheckLinks;
}) {
  const split = useMemo(() => splitIssues(findings.items), [findings.items]);
  const latest = useMemo(
    () =>
      [...split.issues, ...split.possible]
        .sort(newestFirst)
        .slice(0, LISTED_ISSUES),
    [split],
  );
  const any = findings.items.length > 0;
  let status: ReactNode;
  // Drafts and checks being deleted are not read (useCheckFindings).
  if (!findingsReadable(audit) && !any)
    status = (
      <p className="checks-quiet">
        {audit.state === "draft"
          ? "A draft has no issues or possible issues yet."
          : "This check is being deleted; its issues are no longer read."}
      </p>
    );
  else if (findings.isPending)
    status = (
      <p className="checks-quiet" role="status">
        Loading issues…
      </p>
    );
  else if (findings.error !== null && !any)
    status = (
      <div className="checks-notice" data-tone="warning" role="status">
        <p>The issues of this check could not be loaded.</p>
        <button
          className="ui-btn"
          data-size="xs"
          type="button"
          disabled={findings.isFetching}
          onClick={() => void findings.refetch()}
        >
          Try again
        </button>
      </div>
    );
  else
    status = (
      <p className="checks-quiet">
        {issuesSentence(
          findings.items,
          findings.truncated,
          split.setAside.length,
        )}
      </p>
    );
  return (
    <Block title="Issues and possible issues">
      {status}
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
      {any ? (
        <p className="checks-links">
          <Link to={links.section("findings")}>
            All issues and possible issues of this check →
          </Link>
        </p>
      ) : null}
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
  /** Every issue and possible issue of the check (useCheckFindings). */
  findings: AuditCollectionQuery<AuditFinding>;
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
        findings: findings.items,
        waiting: waiting.data?.items ?? [],
        now,
      }),
    [audit, entries, findings.items, now, waiting.data],
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
        <Issues audit={audit} findings={findings} links={links} />
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
      {/* A "#technical-details" link reopens only this disclosure; the
          decisions above keep what the user typed. */}
      <CheckTechnicalDetails
        key={technicalOpen ? "open" : "closed"}
        audit={audit}
        workspace={workspace}
        open={technicalOpen}
      />
    </div>
  );
}
