import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef } from "react";
import { useSearchParams } from "react-router";

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
import { Icon } from "../../app/icon";
import { QueryView } from "../../app/query-view";
import { capitalize, findingStateLabel, TERMS } from "../../app/vocabulary";
import { Button, DetailPane, IconButton } from "../../ui";
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
import { TabPanel, Tabs } from "./tabs";

import { FindingDuplicates } from "./duplicates";

import "./issues.css";

type IssueTab = "summary" | "evidence" | "history";

/** Where the selected possible issue sits in the list. */
interface IssuePosition {
  /** Zero-based. */
  index: number;
  count: number;
}

interface IssueDetailProps {
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
}: IssueDetailProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const idBase = useId();
  const provenanceHeading = useId();
  const header = useRef<HTMLElement>(null);
  const [params, setParams] = useSearchParams();
  const selectedTab = params.get("tab");
  const tab: IssueTab =
    selectedTab === "evidence" || selectedTab === "history"
      ? selectedTab
      : "summary";
  const setTab = (value: IssueTab) => {
    const next = new URLSearchParams(params);
    if (value === "summary") next.delete("tab");
    else next.set("tab", value);
    void setParams(next, { replace: true });
  };

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
            <Icon name="chevron-right" size={16} />
          </ContextLink>
        )}
        <IconButton
          className="issues-step"
          size="sm"
          label="Previous possible issue"
          aria-keyshortcuts="K"
          icon={<Icon name="chevron-up" size={16} />}
          disabled={onPrevious === undefined}
          onClick={onPrevious}
        />
        <IconButton
          className="issues-step"
          size="sm"
          label="Next possible issue"
          aria-keyshortcuts="J"
          icon={<Icon name="chevron-down" size={16} />}
          disabled={onNext === undefined}
          onClick={onNext}
        />
      </div>
    </header>
  );

  const refresh = () =>
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(auditId),
    });
  const blocked = stale || reviewFailed;

  return (
    <DetailPane
      header={bar}
      footer={
        shown === undefined || blocked || waitingForReview ? undefined : (
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
      <QueryView
        query={finding}
        loading={
          <p className="ui-caption" role="status">
            Loading the possible issue…
          </p>
        }
        errorContext="This possible issue could not be loaded."
      >
        {(shown) => (
          <>
            {!blocked ? null : (
              <div className="issues-notice" data-tone="warning" role="alert">
                <p>
                  {reviewFailed
                    ? `The review this link names could not be loaded: ${review.error?.message ?? "unknown error"}.`
                    : "The review this link names no longer matches this possible issue. Refresh the context before deciding, or decide on its current version."}
                </p>
                <div className="issues-notice-actions">
                  <Button size="xs" onClick={refresh}>
                    Refresh context
                  </Button>
                  <Button size="xs" onClick={onDropReview}>
                    Decide on the current version
                  </Button>
                </div>
              </div>
            )}
            <Tabs
              label="Review sections"
              items={[
                { id: "summary", label: "Summary" },
                {
                  id: "evidence",
                  label: "Evidence",
                  count: evidenceCount(shown),
                },
                { id: "history", label: "History" },
              ]}
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
                      <FindingDuplicates
                        key={`${auditId}:${findingId}`}
                        audit={auditShown}
                        finding={shown}
                      />
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
                    <p className="ui-caption">
                      The proposal and the check attempts behind this possible
                      issue.
                    </p>
                    <QueryView
                      query={audit}
                      loading={
                        <p className="ui-caption" role="status">
                          Loading the check…
                        </p>
                      }
                      errorContext="The check could not be loaded."
                    >
                      {(checked) => (
                        <FindingProvenance audit={checked} finding={shown} />
                      )}
                    </QueryView>
                  </section>
                </>
              )}
            </TabPanel>
          </>
        )}
      </QueryView>
    </DetailPane>
  );
}
