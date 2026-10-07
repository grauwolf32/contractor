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
   * undefined when there is nothing to decide or show here.
   */
  review: AuditReviewRequest | undefined;
  /** The report the decision is made next to (ReportDecision's `report`). */
  report: AuditReport | undefined;
  /**
   * The URL names a review (`?review=`) that this report does not carry:
   * the report cannot be used to decide it.
   */
  mismatch: boolean;
  /**
   * The report stopped carrying the request this page showed and reading
   * the request failed: nothing is offered until `retry` reads it.
   */
  error: Error | null;
  /** Reads the request again after `error`. */
  retry: () => void;
  /** That read is running. */
  retrying: boolean;
  /** Pass to ReportDecision: keeps the recorded decision on the page. */
  onDecided: (result: AuditActionDecisionResult) => void;
  /** Pass to ReportDecision: keeps a decision that is recording on the page. */
  onRecording: (recording: boolean) => void;
}

/** The page shows the acceptance part: a request, or why it cannot be read. */
export function showsAcceptance(acceptance: ReportAcceptance): boolean {
  return acceptance.review !== undefined || acceptance.error !== null;
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
 * request). The page then shows what the request says now: the answer of a
 * decision recorded here, else a read of the request. Until that read
 * answers nothing is offered, and when it fails `error` says so, so a
 * request the Server may have settled is never offered next to a report
 * that moved on. Nothing reads as accepted before the Server says so: the
 * report's own status decides what the page shows.
 *
 * A decision recorded on this page stays mounted through that change, so
 * its "Decision recorded" announcement and its focus stay: ReportDecision
 * reports (`onRecording`) while it records, until the refreshed reads
 * arrive and `onDecided` runs, and meanwhile the page keeps offering the
 * request it showed. ReportDecision keeps its bar while it records, even
 * next to the report that moved on.
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
  // A decision of this page is recording (ReportDecision's onRecording).
  const [recording, setRecording] = useState(false);
  const mismatch = requestedReview !== null && requestId !== requestedReview;
  const base = {
    report,
    mismatch,
    error: null,
    retry: () => void read.refetch(),
    retrying: read.isFetching,
    onDecided: (result: AuditActionDecisionResult) =>
      setDecided(result.request),
    onRecording: setRecording,
  };
  if (requestId === undefined || mismatch)
    return { ...base, review: undefined };
  if (carriedNow)
    return {
      ...base,
      review: newestRevision(
        [report?.review, decided, read.data, shown?.review],
        requestId,
      ),
    };
  // The report moved on: show what the request says now, from this page's
  // decision or from the read of the request.
  if (newestRevision([decided, read.data], requestId) !== undefined)
    return {
      ...base,
      review: newestRevision([decided, read.data, shown?.review], requestId),
    };
  // Until then, a decision of this page that is recording stays.
  if (recording) return { ...base, review: shown?.review };
  // Decided elsewhere, or moved on for another reason: nothing to offer
  // until the request is read.
  return { ...base, review: undefined, error: read.error };
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
