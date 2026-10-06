import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, type ReactNode } from "react";
import { Link, useLocation } from "react-router";

import {
  getAuditFinding,
  getAuditReview,
  listAuditFindings,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { EmptyState, FilterChips, ListSection } from "../../../ui";
import { IssueRow } from "../../issues/issue-row";
import { issueHref, type StateFilter } from "../../issues/links";
import { AuditFindingCard } from "./finding-card";
import {
  DECISION_OPTIONS,
  SEVERITY_OPTIONS,
  STATE_OPTIONS,
  isFindingSeverity,
  isFindingState,
  isVerdictFilter,
} from "./finding-options";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";
import { AuditAnchor } from "./shared";

import "../../issues/issues.css";

const STATE_CHIPS: readonly { value: StateFilter; label: string }[] = [
  ...STATE_OPTIONS.map((option) => ({
    value: option.value as StateFilter,
    label: option.label,
  })),
  { value: "all", label: "All" },
];

interface ListLocation {
  to: { pathname: string; search: string };
  /** The page's navigation state, so its return link survives. */
  state: unknown;
}

/** The section's own URL without the deep link to one possible issue. */
function useListLocation(): ListLocation {
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  params.delete("finding");
  params.delete("review");
  const search = params.toString();
  return {
    to: {
      pathname: location.pathname,
      search: search === "" ? "" : `?${search}`,
    },
    state: location.state,
  };
}

function SectionHead({
  audit,
  headingId,
  back,
}: {
  audit: Audit;
  headingId: string;
  back?: ListLocation | undefined;
}) {
  return (
    <header className="issues-section-head">
      <h2 id={headingId} className="issues-section-title">
        Possible issues
      </h2>
      <div className="issues-section-links">
        {back === undefined ? null : (
          <Link to={back.to} state={back.state}>
            All possible issues in this check
          </Link>
        )}
        <ContextLink
          returnLabel="Check possible issues"
          to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}
        >
          All possible issues in this project
        </ContextLink>
      </div>
    </header>
  );
}

/**
 * One possible issue in full inside its check (`?finding=`, optionally with
 * the `?review=` it was opened for). A review that no longer matches the
 * possible issue's revision blocks deciding until the context is refreshed.
 */
function ExactFinding({
  audit,
  findingId,
  reviewId,
}: {
  audit: Audit;
  findingId: string;
  reviewId: string | null;
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const headingId = useId();
  const back = useListLocation();
  const finding = useQuery({
    queryKey: queryKeys.issues.finding(audit.auditId, findingId),
    queryFn: () => getAuditFinding(api, audit.auditId, findingId),
  });
  const review = useQuery({
    queryKey: queryKeys.issues.review(audit.auditId, reviewId ?? ""),
    queryFn: () => getAuditReview(api, audit.auditId, reviewId ?? ""),
    enabled: reviewId !== null,
  });
  const refresh = () =>
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  const head = <SectionHead audit={audit} headingId={headingId} back={back} />;
  let content: ReactNode;
  if (finding.error !== null || review.error !== null)
    content = (
      <AuditQueueError
        error={(finding.error ?? review.error) as Error}
        onRefresh={refresh}
      />
    );
  else if (finding.isPending || (reviewId !== null && review.isPending))
    content = (
      <p className="issues-quiet" role="status">
        Loading the possible issue and its review…
      </p>
    );
  else {
    const requested = review.data;
    const stale =
      requested !== undefined &&
      (requested.subjectKind !== "finding" ||
        requested.findingId !== findingId ||
        (requested.state === "pending" &&
          requested.subjectRevision !== finding.data.revision));
    content = (
      <>
        {stale ? (
          <div className="issues-notice" data-tone="warning" role="alert">
            <p>
              The requested review no longer matches this possible issue&apos;s
              current version. Refresh the context before making a decision.
            </p>
            <div className="issues-notice-actions">
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                onClick={refresh}
              >
                Refresh context
              </button>
            </div>
          </div>
        ) : null}
        <p className="issues-section-note">
          <Link
            to={issueHref(finding.data, {
              state: "all",
              project: audit.projectId,
            })}
          >
            Open in Issues
          </Link>
        </p>
        <AuditFindingCard
          audit={audit}
          finding={finding.data}
          pendingReview={
            !stale && requested?.state === "pending" ? requested : undefined
          }
          reviewError={
            stale
              ? new Error(
                  "Deciding is closed until the context is refreshed: the requested review changed.",
                )
              : null
          }
        />
      </>
    );
  }
  return (
    <section className="issues-section" aria-labelledby={headingId}>
      <AuditAnchor ready={finding.data !== undefined} />
      {head}
      {content}
    </section>
  );
}

/**
 * The check's possible issues, filtered on the Server by state, decision
 * and the analyst's severity, one page at a time pinned to the check's
 * revision. Each row opens the possible issue on the Issues destination;
 * `?finding=` shows one in full here.
 */
export function AuditFindings({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const headingId = useId();
  const decisionId = useId();
  const severityId = useId();
  const queue = useAuditQueue();
  const rawState = queue.params.get("state") ?? "";
  const rawVerdict = queue.params.get("verdict") ?? "";
  const rawSeverity = queue.params.get("severity") ?? "";
  const state = isFindingState(rawState) ? rawState : undefined;
  const verdict = isVerdictFilter(rawVerdict) ? rawVerdict : undefined;
  const severity = isFindingSeverity(rawSeverity) ? rawSeverity : undefined;
  const exactId = queue.params.get("finding");
  const findings = useQuery({
    queryKey: [
      ...queryKeys.audits.allFindings(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditFindings(api, audit.auditId, {
        ...queue.request,
        ...(state === undefined ? {} : { state }),
        ...(verdict === undefined ? {} : { verdict }),
        ...(severity === undefined ? {} : { severity }),
      }),
    enabled: exactId === null,
  });
  const refresh = () => {
    queue.refresh();
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  };
  if (exactId !== null)
    return (
      <ExactFinding
        audit={audit}
        findingId={exactId}
        reviewId={queue.params.get("review")}
      />
    );
  return (
    <section className="issues-section" aria-labelledby={headingId}>
      <AuditAnchor ready={findings.data !== undefined} />
      <SectionHead audit={audit} headingId={headingId} />
      <div className="issues-section-filters">
        <FilterChips
          label="Filter by state"
          options={STATE_CHIPS}
          value={state ?? "all"}
          onChange={(value) => queue.change("state", value)}
        />
        <div className="issues-filter-row">
          <span className="issues-select-field">
            <label htmlFor={decisionId} className="issues-field-label">
              Decision
            </label>
            <select
              id={decisionId}
              className="issues-select"
              value={verdict ?? ""}
              onChange={(event) => queue.change("verdict", event.target.value)}
            >
              {DECISION_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </span>
          <span className="issues-select-field">
            <label htmlFor={severityId} className="issues-field-label">
              Severity
            </label>
            <select
              id={severityId}
              className="issues-select"
              aria-describedby={`${severityId}-hint`}
              value={severity ?? ""}
              onChange={(event) => queue.change("severity", event.target.value)}
            >
              {SEVERITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </span>
        </div>
        <p id={`${severityId}-hint`} className="issues-quiet">
          Severity is the analyst&apos;s rating. The AI&apos;s suggestion stays
          with each possible issue and is never filtered on.
        </p>
      </div>
      {findings.isPending ? (
        <p className="issues-quiet" role="status">
          Loading possible issues…
        </p>
      ) : findings.error !== null ? (
        <AuditQueueError error={findings.error} onRefresh={refresh} />
      ) : (
        <>
          <AuditQueuePage
            page={findings.data}
            currentRevision={audit.revision}
            queue={queue}
            onRefresh={refresh}
          />
          {findings.data.items.length === 0 ? (
            <div className="issues-embedded-empty">
              <EmptyState title="No matching possible issues">
                <p>
                  A successful Run alone does not create or confirm a possible
                  issue.
                </p>
              </EmptyState>
            </div>
          ) : (
            <div className="issues-embedded-list">
              <ListSection>
                {findings.data.items.map((finding) => (
                  <IssueRow
                    key={finding.findingId}
                    id={`finding-${audit.auditId}-${finding.findingId}`}
                    finding={finding}
                    to={issueHref(finding, {
                      state: state ?? "all",
                      project: audit.projectId,
                      severity,
                    })}
                  />
                ))}
              </ListSection>
            </div>
          )}
        </>
      )}
    </section>
  );
}
