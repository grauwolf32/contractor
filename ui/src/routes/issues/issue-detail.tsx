import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import { Link, type To } from "react-router";

import {
  getAudit,
  getAuditFinding,
  getAuditReview,
  listAuditReviews,
  type AuditFinding,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import type { CrossProjectIssue } from "../../api/cross-project";
import { getProject, type Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { capitalize, findingStateLabel, TERMS } from "../../app/vocabulary";
import { DetailPane } from "../../ui";
import {
  FindingDecision,
  FindingSummary,
  type AuditFindingDecisionResult,
} from "../decisions";
import {
  FindingProvenance,
  FindingSources,
  FindingTechnicalDetails,
} from "../projects/audits/finding-card";
import { auditProfileLabel } from "../projects/audits/labels";
import { evidenceCount } from "./evidence";
import { EvidencePanel } from "./evidence-panel";
import { DecisionHistory } from "./history";
import { checkIssuePath } from "./links";
import { TabPanel, Tabs, type TabItem } from "./tabs";

import "./issues.css";

type IssueTab = "summary" | "evidence" | "history";

function Icon({ path }: { path: string }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={path} />
    </svg>
  );
}

const CHEVRON_RIGHT = "M9.5 6l6 6-6 6";
const CHEVRON_UP = "M6 14.5l6-6 6 6";
const CHEVRON_DOWN = "M6 9.5l6 6 6-6";

/** Where the selected possible issue sits in the list. */
export interface IssuePosition {
  /** Zero-based. */
  index: number;
  count: number;
}

export interface IssueDetailProps {
  auditId: string;
  findingId: string;
  /** The possible issue as the list has it, shown until the exact read. */
  listed?: CrossProjectIssue | undefined;
  /** Projects already read, to name the possible issue's project. */
  projects: readonly Project[];
  /** Undefined when the possible issue is not in the filtered list. */
  position?: IssuePosition | undefined;
  /**
   * Some list read has not settled: a possible issue without a position
   * may still arrive (a deep link's exact read usually finishes first).
   */
  listPending?: boolean | undefined;
  onPrevious?: (() => void) | undefined;
  onNext?: (() => void) | undefined;
  /** A review request named by the link (`?review=`). */
  reviewId: string | null;
  /** Drops `?review=` to decide on the current version. */
  onDropReview: () => void;
  onDecided: (result: AuditFindingDecisionResult) => void;
  /** Changes when focus should move to the top of the detail. */
  focusRequest?: number | undefined;
  backTo: To;
}

function PanelTitle({ finding }: { finding: AuditFinding }) {
  return (
    <h2 className="issues-panel-title">
      {finding.firstProposal.document.title}
    </h2>
  );
}

/**
 * One possible issue: a header bar with its place in the list, the check
 * it belongs to and previous / next; tabs Summary, Evidence and History;
 * and the decision pinned to the bottom.
 */
export function IssueDetail({
  auditId,
  findingId,
  listed,
  projects,
  position,
  listPending = false,
  onPrevious,
  onNext,
  reviewId,
  onDropReview,
  onDecided,
  focusRequest,
  backTo,
}: IssueDetailProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const idBase = useId();
  const provenanceHeading = useId();
  const header = useRef<HTMLElement>(null);
  const [tab, setTab] = useState<IssueTab>("summary");

  const finding = useQuery({
    queryKey: queryKeys.issues.finding(auditId, findingId),
    queryFn: () => getAuditFinding(api, auditId, findingId),
    placeholderData: () => listed?.finding,
    retry: false,
  });
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    queryFn: () => getAudit(api, auditId),
    placeholderData: () => listed?.audit,
    // J / K through one check's possible issues need not re-read it each time.
    staleTime: 15_000,
    retry: false,
  });
  const projectId = audit.data?.projectId ?? listed?.project.projectId;
  const known =
    projects.find((candidate) => candidate.projectId === projectId) ??
    (listed?.project.projectId === projectId ? listed?.project : undefined);
  const projectRead = useQuery({
    queryKey: queryKeys.projects.detail(projectId ?? ""),
    queryFn: () => getProject(api, projectId ?? ""),
    enabled: projectId !== undefined && known === undefined,
    retry: false,
  });
  const project = known ?? projectRead.data;
  const review = useQuery({
    queryKey: queryKeys.issues.review(auditId, reviewId ?? ""),
    queryFn: () => getAuditReview(api, auditId, reviewId ?? ""),
    enabled: reviewId !== null,
    retry: false,
  });
  useEffect(() => {
    if (focusRequest !== undefined) header.current?.focus();
  }, [focusRequest]);

  // The polled list saw a newer revision (someone decided, the check
  // assessed it again): read the possible issue and its check again.
  const listedFinding = listed?.finding.revision;
  const shownFinding = finding.data?.revision;
  const listedAudit = listed?.audit.revision;
  const shownAudit = audit.data?.revision;
  useEffect(() => {
    if (
      listedFinding !== undefined &&
      shownFinding !== undefined &&
      listedFinding > shownFinding
    )
      void queryClient.invalidateQueries({
        queryKey: queryKeys.issues.finding(auditId, findingId),
        exact: true,
      });
  }, [auditId, findingId, listedFinding, queryClient, shownFinding]);
  useEffect(() => {
    if (
      listedAudit !== undefined &&
      shownAudit !== undefined &&
      listedAudit > shownAudit
    )
      void queryClient.invalidateQueries({
        queryKey: queryKeys.audits.detail(auditId),
        exact: true,
      });
  }, [auditId, listedAudit, queryClient, shownAudit]);

  const shown = finding.data;
  const exact = shown !== undefined && !finding.isPlaceholderData;
  const requested = reviewId === null ? undefined : review.data;
  // A review the link names that no longer fits this possible issue: the
  // decision stays closed until the user refreshes or drops the link's review.
  const stale =
    exact &&
    requested !== undefined &&
    (requested.subjectKind !== "finding" ||
      (requested.findingId ?? requested.subjectId) !== findingId ||
      (requested.state === "pending" &&
        requested.subjectRevision !== shown.revision));
  const reviewFailed = reviewId !== null && review.error !== null;
  const pendingReview =
    requested !== undefined && !stale && requested.state === "pending"
      ? requested
      : undefined;
  const waitingForReview =
    reviewId !== null && review.data === undefined && review.error === null;
  // The open review requests FindingDecision reads when the link names none
  // (same key and request, so no second read): their ID belongs in the
  // technical details.
  const openReviews = useQuery({
    queryKey: [...queryKeys.audits.reviews(auditId, findingId), "pending"],
    queryFn: () =>
      listAuditReviews(api, auditId, { finding: findingId, state: "pending" }),
    enabled: pendingReview === undefined && !waitingForReview,
  });
  const openReview =
    pendingReview ??
    openReviews.data?.items.find(
      (candidate) =>
        candidate.state === "pending" &&
        candidate.kind === "finding-triage" &&
        (candidate.findingId ?? candidate.subjectId) === findingId &&
        candidate.subjectRevision === shown?.revision,
    );

  const state =
    shown === undefined ? undefined : findingStateLabel(shown.state);
  const kind =
    shown?.state === "confirmed"
      ? capitalize(TERMS.issue)
      : capitalize(TERMS.possibleIssue);
  const auditShown = audit.data;
  const context = [
    project?.name,
    auditShown === undefined ? undefined : auditProfileLabel(auditShown),
  ].filter((part) => part !== undefined);

  const bar = (
    <header ref={header} tabIndex={-1} className="issues-detail-bar">
      <p className="issues-detail-position">
        <strong>
          {position === undefined
            ? kind
            : `${kind} ${position.index + 1} of ${position.count}`}
        </strong>
        {state === undefined ? null : <span>{state.label}</span>}
        {position === undefined && shown !== undefined ? (
          <span>{listPending ? "Loading the list…" : "Not in this list"}</span>
        ) : null}
        {context.length === 0 ? null : (
          <span className="issues-detail-context">{context.join(" · ")}</span>
        )}
      </p>
      <div className="issues-detail-nav">
        {projectId === undefined ? null : (
          <ContextLink
            returnLabel="Possible issue"
            className="issues-detail-check"
            to={checkIssuePath(projectId, auditId, findingId)}
          >
            View in its check
            <Icon path={CHEVRON_RIGHT} />
          </ContextLink>
        )}
        <button
          type="button"
          className="ui-btn issues-step"
          data-size="sm"
          aria-label="Previous possible issue"
          aria-keyshortcuts="K"
          title="Previous (K)"
          disabled={onPrevious === undefined}
          onClick={onPrevious}
        >
          <Icon path={CHEVRON_UP} />
        </button>
        <button
          type="button"
          className="ui-btn issues-step"
          data-size="sm"
          aria-label="Next possible issue"
          aria-keyshortcuts="J"
          title="Next (J)"
          disabled={onNext === undefined}
          onClick={onNext}
        >
          <Icon path={CHEVRON_DOWN} />
        </button>
      </div>
    </header>
  );

  if (shown === undefined)
    return (
      <DetailPane header={bar}>
        {finding.error === null ? (
          <p className="issues-quiet" role="status">
            Loading the possible issue…
          </p>
        ) : (
          <div className="issues-notice" data-tone="error" role="alert">
            <p>
              <strong>This possible issue could not be loaded.</strong>{" "}
              {finding.error.message}
            </p>
            <div className="issues-notice-actions">
              <button
                type="button"
                className="ui-btn"
                data-size="xs"
                disabled={finding.isFetching}
                onClick={() => void finding.refetch()}
              >
                Try again
              </button>
              <Link to={backTo}>Back to possible issues</Link>
            </div>
          </div>
        )}
      </DetailPane>
    );

  const refresh = () =>
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(auditId),
    });
  const tabs: TabItem<IssueTab>[] = [
    { id: "summary", label: "Summary" },
    { id: "evidence", label: "Evidence", count: evidenceCount(shown) },
    { id: "history", label: "History" },
  ];
  const blocked = stale || reviewFailed;

  return (
    <DetailPane
      header={bar}
      footer={
        blocked || waitingForReview ? undefined : (
          <FindingDecision
            auditId={auditId}
            finding={shown}
            pendingReview={pendingReview}
            next={
              onNext === undefined ? undefined : { label: "Next item", onNext }
            }
            onDecided={onDecided}
          />
        )
      }
    >
      {!blocked ? null : (
        <div className="issues-notice" data-tone="warning" role="alert">
          <p>
            {reviewFailed
              ? `The review this link names could not be loaded: ${review.error?.message ?? "unknown error"}.`
              : "The review this link names no longer matches this possible issue. Refresh the context before deciding, or decide on its current version."}
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
            <button
              type="button"
              className="ui-btn"
              data-size="xs"
              onClick={onDropReview}
            >
              Decide on the current version
            </button>
          </div>
        </div>
      )}
      <Tabs
        label="Review sections"
        items={tabs}
        selected={tab}
        onSelect={setTab}
        idBase={idBase}
      />
      <TabPanel idBase={idBase} id={tab}>
        {tab === "summary" ? (
          <>
            <FindingSummary auditId={auditId} finding={shown} />
            {auditShown === undefined ? null : (
              <>
                <FindingSources
                  audit={auditShown}
                  returnLabel="Possible issue"
                />
                <FindingTechnicalDetails
                  audit={auditShown}
                  finding={shown}
                  pendingReview={openReview}
                  showCheckId
                />
              </>
            )}
          </>
        ) : tab === "evidence" ? (
          <>
            <PanelTitle finding={shown} />
            <EvidencePanel finding={shown} />
          </>
        ) : (
          <>
            <PanelTitle finding={shown} />
            <DecisionHistory auditId={auditId} findingId={findingId} />
            <section
              className="issues-history"
              aria-labelledby={provenanceHeading}
            >
              <h3 id={provenanceHeading} className="issues-heading">
                Provenance
              </h3>
              <p className="issues-quiet">
                The proposal and the check attempts behind this possible issue.
              </p>
              {auditShown === undefined ? (
                audit.error === null ? (
                  <p className="issues-quiet" role="status">
                    Loading the check…
                  </p>
                ) : (
                  <div className="issues-notice" data-tone="error" role="alert">
                    <p>The check could not be loaded: {audit.error.message}</p>
                    <div className="issues-notice-actions">
                      <button
                        type="button"
                        className="ui-btn"
                        data-size="xs"
                        disabled={audit.isFetching}
                        onClick={() => void audit.refetch()}
                      >
                        Try again
                      </button>
                    </div>
                  </div>
                )
              ) : (
                <FindingProvenance audit={auditShown} finding={shown} />
              )}
            </section>
          </>
        )}
      </TabPanel>
    </DetailPane>
  );
}
