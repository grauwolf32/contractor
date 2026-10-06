import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { useState } from "react";

import {
  AUDIT_ID_PATTERN,
  auditPollInterval,
  getAudit,
  getAuditReport,
  getAuditReview,
  type Audit,
  type AuditActionDecisionResult,
  type AuditReport,
  type AuditReviewRequest,
} from "../../api/audits";
import type { PublicAPI } from "../../api/client";
import { queryKeys } from "../../api/query-keys";
import { saveBlob } from "../../app/download";
import { useAuditProjectionRefresh } from "../projects/audits/projection-refresh";

/**
 * The selected check of /reports/:auditId, read with the check page's key
 * and polled like it while the check can change. Idle without a valid ID.
 */
export function useReportCheck(
  api: PublicAPI,
  auditId: string | undefined,
): UseQueryResult<Audit> {
  return useQuery({
    queryKey: queryKeys.audits.detail(auditId ?? ""),
    queryFn: () => getAudit(api, auditId ?? ""),
    enabled: auditId !== undefined && AUDIT_ID_PATTERN.test(auditId),
    refetchInterval: (query) =>
      auditPollInterval(
        query.state.data === undefined ? [] : [query.state.data],
        1_000,
      ),
    refetchOnReconnect: true,
  });
}

/**
 * The check's report, read with the same key as the check page and the
 * decision components. It is polled while the check can still change, and
 * read once more when polling stops.
 */
export function useAuditReport(
  audit: Audit,
  api: PublicAPI,
): UseQueryResult<AuditReport> {
  const queryKey = queryKeys.audits.report(audit.auditId);
  const report = useQuery({
    queryKey,
    queryFn: () => getAuditReport(api, audit.auditId),
    refetchInterval: auditPollInterval([audit], 1_000),
    refetchOnReconnect: true,
  });
  useAuditProjectionRefresh(audit, queryKey);
  return report;
}

/** What a report page offers for the report's acceptance request. */
export interface ReportAcceptance {
  /**
   * The acceptance request to decide, or the decided one this page showed;
   * undefined when there is nothing to decide here.
   */
  review: AuditReviewRequest | undefined;
  /** The report the decision is made next to (ReportDecision's `report`). */
  report: AuditReport | undefined;
  /**
   * The URL names a review (`?review=`) that this report does not carry:
   * the report cannot be used to decide it.
   */
  mismatch: boolean;
  /** Pass to ReportDecision: keeps the recorded decision on the page. */
  onDecided: (result: AuditActionDecisionResult) => void;
}

function newestRevision(
  candidates: readonly (AuditReviewRequest | undefined)[],
  requestId: string,
): AuditReviewRequest | undefined {
  let newest: AuditReviewRequest | undefined;
  for (const candidate of candidates) {
    if (candidate === undefined || candidate.requestId !== requestId) continue;
    if (newest === undefined || candidate.revision > newest.revision)
      newest = candidate;
  }
  return newest;
}

/**
 * The acceptance request of a proposed report, kept on the page once shown.
 *
 * A proposed report carries its request; once the request is decided the
 * check moves on and the report no longer carries it (a ready report has no
 * request). The page keeps the decision mounted through that change, so the
 * recorded decision and its announcement stay: the request's own state comes
 * from the decision's answer and from a read of the request, and while that
 * read loads the decision stays next to the report it was opened on. Nothing
 * reads as accepted before the Server says so: the report's own status
 * decides what the page shows.
 *
 * `requestedReview` is the `?review=` of review links: only that request is
 * offered, and a report that does not carry it is a mismatch.
 *
 * Key the caller by check, so the kept request never crosses checks.
 */
export function useReportAcceptance(
  api: PublicAPI,
  auditId: string,
  report: AuditReport | undefined,
  requestedReview: string | null,
): ReportAcceptance {
  const carried =
    report?.status === "proposed" &&
    report.review?.kind === "report-acceptance" &&
    (requestedReview === null || report.review.requestId === requestedReview)
      ? report
      : undefined;
  // The last proposed report that carried a request on this page.
  const [pinned, setPinned] = useState<AuditReport | undefined>(undefined);
  if (carried !== undefined && carried !== pinned) setPinned(carried);
  const shown = carried ?? pinned;
  const requestId = shown?.review?.requestId;
  // The request as the recorded decision returned it.
  const [decided, setDecided] = useState<AuditReviewRequest | undefined>(
    undefined,
  );
  const carriedNow =
    requestId !== undefined && report?.review?.requestId === requestId;
  // Once the report stops carrying the request, read the request itself.
  const read = useQuery({
    queryKey: queryKeys.reports.acceptance(auditId, requestId ?? ""),
    queryFn: () => getAuditReview(api, auditId, requestId ?? ""),
    enabled: requestId !== undefined && !carriedNow,
  });
  const mismatch = requestedReview !== null && requestId !== requestedReview;
  const onDecided = (result: AuditActionDecisionResult) =>
    setDecided(result.request);
  if (requestId === undefined || mismatch)
    return { review: undefined, report, mismatch, onDecided };
  const review = newestRevision(
    [
      carriedNow ? report?.review : undefined,
      decided,
      read.data,
      shown?.review,
    ],
    requestId,
  );
  const reading = !carriedNow && (read.isPending || read.isFetching);
  return {
    review,
    report: reading ? shown : report,
    mismatch,
    onDecided,
  };
}

const CERTIFICATION_NOTICE = "not a security or compliance certification";

/** The summary already says the report is not a certification. */
export function summaryDisclaimsCertification(
  summary: string | undefined,
): boolean {
  return (
    summary !== undefined &&
    summary.replace(/\s+/g, " ").toLowerCase().includes(CERTIFICATION_NOTICE)
  );
}

/** Saves report content generated in the browser from the report response. */
export function downloadReportFile(
  name: string,
  mediaType: string,
  content: string,
): void {
  saveBlob(new Blob([content], { type: mediaType }), name);
}
