import { useQuery, useQueryClient } from "@tanstack/react-query";

import {
  getAuditFinding,
  getAuditReview,
  listAuditFindings,
  listAuditReviews,
  type Audit,
  type AuditFinding,
  type AuditFindingSeverity,
  type AuditFindingState,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { queryKeys } from "../../../api/query-keys";
import { ContextLink } from "../../../app/context-navigation";
import { AuditFindingCard } from "./finding-card";
import { FINDING_SEVERITIES } from "./finding-options";
import { AuditQueueError, AuditQueuePage } from "./queue";
import { useAuditQueue } from "./queue-state";
import { AuditAnchor } from "./shared";

function FindingInQueue({
  audit,
  finding,
  siblings,
}: {
  audit: Audit;
  finding: AuditFinding;
  siblings: AuditFinding[];
}) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const reviews = useQuery({
    queryKey: [
      ...queryKeys.audits.reviews(audit.auditId, finding.findingId),
      "pending",
    ],
    queryFn: () =>
      listAuditReviews(api, audit.auditId, {
        finding: finding.findingId,
        state: "pending",
      }),
  });
  const pending = reviews.data?.items.find(
    (review) => review.state === "pending",
  );
  return (
    <AuditFindingCard
      audit={audit}
      finding={finding}
      findings={siblings}
      {...(pending === undefined ? {} : { pendingReview: pending })}
      reviewLoading={reviews.isPending}
      reviewError={
        reviews.error ??
        (pending !== undefined && pending.subjectRevision !== finding.revision
          ? new Error(
              "Finding evidence changed. Refresh the context before deciding.",
            )
          : null)
      }
      onRetryReview={() =>
        void queryClient.invalidateQueries({
          queryKey: queryKeys.audits.detail(audit.auditId),
        })
      }
    />
  );
}

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
  const finding = useQuery({
    queryKey: [
      ...queryKeys.audits.detail(audit.auditId),
      "findings",
      findingId,
      "exact",
    ],
    queryFn: () => getAuditFinding(api, audit.auditId, findingId),
  });
  const review = useQuery({
    queryKey: [...queryKeys.audits.detail(audit.auditId), "reviews", reviewId],
    queryFn: () => getAuditReview(api, audit.auditId, reviewId!),
    enabled: reviewId !== null,
  });
  const refresh = () =>
    void queryClient.invalidateQueries({
      queryKey: queryKeys.audits.detail(audit.auditId),
    });
  if (finding.error !== null || review.error !== null)
    return (
      <AuditQueueError
        error={(finding.error ?? review.error)!}
        onRefresh={refresh}
      />
    );
  if (finding.isPending || (reviewId !== null && review.isPending))
    return <p role="status">Loading exact finding and review…</p>;
  const stale =
    review.data !== undefined &&
    (review.data.subjectKind !== "finding" ||
      review.data.findingId !== findingId ||
      (review.data.state === "pending" &&
        review.data.subjectRevision !== finding.data.revision));
  return (
    <div className="audit-finding-list">
      <AuditAnchor />
      {stale ? (
        <div className="notice notice-error">
          The requested review no longer matches this finding revision. Refresh
          the context before making a decision.
          <button type="button" className="secondary-button" onClick={refresh}>
            Refresh context
          </button>
        </div>
      ) : null}
      {stale || review.data?.state === "pending" ? (
        <AuditFindingCard
          audit={audit}
          finding={finding.data}
          findings={[finding.data]}
          {...(!stale && review.data?.state === "pending"
            ? { pendingReview: review.data }
            : {})}
          reviewError={
            stale ? new Error("Review subject revision changed") : null
          }
        />
      ) : null}
      {!stale && review.data?.state !== "pending" ? (
        <FindingInQueue
          audit={audit}
          finding={finding.data}
          siblings={[finding.data]}
        />
      ) : null}
    </div>
  );
}

export function AuditFindings({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const queue = useAuditQueue();
  const state = queue.params.get("state") ?? "";
  const verdict = queue.params.get("verdict") ?? "";
  const severity = queue.params.get("severity") ?? "";
  const exactId = queue.params.get("finding");
  const findings = useQuery({
    queryKey: [
      ...queryKeys.audits.allFindings(audit.auditId),
      queue.params.toString(),
    ],
    queryFn: () =>
      listAuditFindings(api, audit.auditId, {
        ...queue.request,
        ...([
          "proposed",
          "confirmed",
          "rejected",
          "duplicate",
          "needs-evidence",
        ].includes(state)
          ? { state: state as AuditFindingState }
          : {}),
        ...(["unreviewed", "true_positive", "false_positive"].includes(verdict)
          ? {
              verdict: verdict as
                "unreviewed" | "true_positive" | "false_positive",
            }
          : {}),
        ...(FINDING_SEVERITIES.includes(severity as AuditFindingSeverity)
          ? { severity: severity as AuditFindingSeverity }
          : {}),
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
    <section className="audit-finding-list">
      <AuditAnchor />
      <div className="section-heading">
        <h3>Findings in this audit</h3>
        <ContextLink
          returnLabel="Audit findings"
          to={`/projects/${encodeURIComponent(audit.projectId)}/findings`}
        >
          View all project findings →
        </ContextLink>
      </div>
      <div className="audit-review-fields">
        <label>
          Finding disposition
          <select
            value={state}
            onChange={(event) => queue.change("state", event.target.value)}
          >
            <option value="">All dispositions</option>
            {[
              "proposed",
              "confirmed",
              "rejected",
              "duplicate",
              "needs-evidence",
            ].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          Analyst verdict
          <select
            value={verdict}
            onChange={(event) => queue.change("verdict", event.target.value)}
          >
            <option value="">All verdicts</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="true_positive">True positive</option>
            <option value="false_positive">False positive</option>
          </select>
        </label>
        <label>
          Analyst severity
          <select
            value={severity}
            onChange={(event) => queue.change("severity", event.target.value)}
          >
            <option value="">All severities</option>
            {FINDING_SEVERITIES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="muted-copy">
        Severity filters apply to analyst decisions. Model proposals remain
        separate from accepted findings.
      </p>
      {findings.isPending ? (
        <p role="status">Loading findings…</p>
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
            <div className="empty-state panel">
              <h3>No matching finding candidates</h3>
              <p>
                A successful Run alone does not create or confirm a finding.
              </p>
            </div>
          ) : (
            findings.data.items.map((finding) => (
              <FindingInQueue
                key={finding.findingId}
                audit={audit}
                finding={finding}
                siblings={findings.data.items}
              />
            ))
          )}
        </>
      )}
    </section>
  );
}
