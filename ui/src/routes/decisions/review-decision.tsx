import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState, type ReactNode } from "react";

import {
  decideAuditAction,
  getAuditReport,
  type AuditReport,
  type AuditReviewAction,
  type AuditReviewRequest,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { reviewKindLabel } from "../../app/vocabulary";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import { DecisionBar, type DecisionOption } from "../../ui";
import { focusIfLost, useAnnouncement } from "./announcement";
import { DecisionRecord } from "./decision-record";
import {
  decisionErrorMessage,
  decisionOutcome,
  MAX_RATIONALE_BYTES,
  RATIONALE_HELP,
  REVIEW_ACTIONS,
  utf8Length,
  type AuditActionDecisionResult,
} from "./model";
import { useRecordingCallback } from "./recording";
import { refreshAfterDecision } from "./refresh";
import { RequestDetails } from "./request-details";

import "./decisions.css";

function isAction(value: string): value is AuditReviewAction {
  return REVIEW_ACTIONS.some((candidate) => candidate.action === value);
}

interface RequestDecisionProps {
  auditId: string;
  review: AuditReviewRequest;
  onDecided?: ((result: AuditActionDecisionResult) => void) | undefined;
  onRecording?: ((recording: boolean) => void) | undefined;
  /** What the actions do, shown under them. */
  intro?: ReactNode;
  /**
   * Why this pending request cannot be decided here (yet); shown instead of
   * the actions, except while a decision made here is being recorded.
   */
  blocked?: ReactNode;
}

/**
 * Approve / Reject / Not applicable on one review request, offering only the
 * actions the Server requested. A decided request shows its decision, an
 * expired one says so.
 */
function RequestDecision({
  auditId,
  review,
  onDecided,
  onRecording,
  intro,
  blocked,
}: RequestDecisionProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const root = useRef<HTMLDivElement>(null);
  const [action, setAction] = useState<AuditReviewAction | undefined>();
  const [rationale, setRationale] = useState("");
  const [problem, setProblem] = useState<string | undefined>();
  // The request revision a recorded decision was made on: until the
  // refetched request replaces it, nothing is offered.
  const [recordedRevision, setRecordedRevision] = useState<number | null>(null);
  // "Decision recorded: …" outlives that gate, so screen readers read it.
  const announcement = useAnnouncement();
  const [keyring] = useState(
    () =>
      new MutationDraftKeyring<Record<string, string | number>>(
        "audit-action-review",
      ),
  );
  const decide = useMutation({
    mutationFn: (choice: {
      action: AuditReviewAction;
      rationale: string;
      revision: number;
    }) =>
      decideAuditAction(api, {
        auditId,
        requestId: review.requestId,
        expectedRevision: choice.revision,
        idempotencyKey: keyring.keyFor({
          auditId,
          requestId: review.requestId,
          revision: choice.revision,
          action: choice.action,
          rationale: choice.rationale,
        }),
        action: choice.action,
        rationale: choice.rationale,
      }),
    onSuccess: async (result, choice) => {
      await refreshAfterDecision(queryClient, auditId);
      setRecordedRevision(choice.revision);
      announcement.announce(
        `Decision recorded: ${decisionOutcome(result.decision).label}.`,
      );
      setAction(undefined);
      setRationale("");
      // The bar goes away; keep keyboard and screen reader users here.
      focusIfLost(root.current);
      onDecided?.(result);
    },
    onError: () => refreshAfterDecision(queryClient, auditId),
  });

  // Until the refetched request arrives, only the announcement shows.
  const awaitingRefresh =
    recordedRevision !== null && recordedRevision === review.revision;
  // From sending a decision until the refetched request arrives. The reads
  // refresh meanwhile, so the page can see the subject move on (a report
  // that is no longer proposed) before this decision's answer is shown:
  // the bar stays until then instead of turning into `blocked`.
  const recording = decide.isPending || awaitingRefresh;
  useRecordingCallback(recording, onRecording);
  const options: DecisionOption[] = REVIEW_ACTIONS.filter((candidate) =>
    review.requestedActions.includes(candidate.action),
  ).map((candidate) => ({
    id: candidate.action,
    label: candidate.label,
    tone: candidate.action === "approve" ? "primary" : "secondary",
  }));
  const decideMessage =
    decide.error === null
      ? undefined
      : decisionErrorMessage(decide.error, "request");

  // Every edit of the draft: the last answer's announcement and a client-side
  // problem no longer apply.
  function edited() {
    setProblem(undefined);
    announcement.clear();
  }

  function submit() {
    if (action === undefined) return;
    const trimmed = rationale.trim();
    if (utf8Length(trimmed) > MAX_RATIONALE_BYTES) {
      setProblem("The reason is too long. Keep it under 64 KiB.");
      return;
    }
    setProblem(undefined);
    decide.mutate({ action, rationale: trimmed, revision: review.revision });
  }

  let body: ReactNode;
  // A recorded decision is capped where the pane footer is pinned (ui.css).
  let record = false;
  let bar = false;
  if (awaitingRefresh) body = null;
  else if (review.state === "decided") {
    record = review.decision !== undefined;
    body =
      review.decision === undefined ? (
        <p className="decisions-quiet">Decided.</p>
      ) : (
        <DecisionRecord decision={review.decision} />
      );
  } else if (review.state === "expired")
    body = (
      <p className="decisions-quiet">
        This request expired without a decision.
      </p>
    );
  else if (blocked !== undefined && blocked !== null && !recording)
    body = blocked;
  else if (options.length === 0)
    body = (
      <p className="decisions-quiet">
        This request offers no decision that can be made here.
      </p>
    );
  else {
    bar = true;
    body = (
      <>
        <DecisionBar
          options={options}
          selected={action}
          onSelect={(id) => {
            if (isAction(id)) {
              setAction(id);
              edited();
            }
          }}
          rationale={{
            value: rationale,
            onChange: (value) => {
              setRationale(value);
              edited();
            },
            maxLength: MAX_RATIONALE_BYTES,
            hint: RATIONALE_HELP,
          }}
          onSubmit={submit}
          pending={decide.isPending}
          error={problem ?? decideMessage}
          extra={
            <>
              {intro === undefined ? null : (
                <p className="decisions-intro">{intro}</p>
              )}
            </>
          }
        />
        {problem === undefined ? (
          // DecisionBar's error slot is a paragraph; the disclosure follows.
          <RequestDetails
            error={decide.error}
            explanation={decideMessage}
            className="decisions-under-bar"
          />
        ) : null}
      </>
    );
  }

  return (
    <div
      ref={root}
      className={
        record ? "decisions-request ui-footer-record" : "decisions-request"
      }
      role="group"
      aria-label={`Decision on ${reviewKindLabel(review.kind).toLowerCase()}`}
      tabIndex={-1}
    >
      <p className="decisions-status" role="status">
        {announcement.text}
      </p>
      {!bar && decide.error !== null ? (
        // The refetch after a refused decision shows the request settled
        // elsewhere, or no longer decidable here; the explanation stays.
        <div className="decisions-notice decisions-inset" role="alert">
          <p>{decideMessage}</p>
          <RequestDetails error={decide.error} explanation={decideMessage} />
        </div>
      ) : null}
      {body}
    </div>
  );
}

export interface ActionDecisionProps {
  auditId: string;
  /** An active test approval or requirement applicability request. */
  review: AuditReviewRequest;
  /** Called once the Server recorded the decision and the reads refetched. */
  onDecided?: ((result: AuditActionDecisionResult) => void) | undefined;
  /**
   * True once a decision is sent, false once its refetched request arrived
   * (or it was refused and the reads refreshed). Keep the decision mounted
   * meanwhile, even when the page's own reads already moved on.
   */
  onRecording?: ((recording: boolean) => void) | undefined;
}

const ACTION_INTROS: Partial<Record<AuditReviewRequest["kind"], string>> = {
  "active-check-approval":
    "Approving lets this active test run. Rejecting excludes it from the check.",
  "requirement-applicability":
    "Not applicable settles this requirement with your reason and removes it from the coverage count.",
};

/**
 * Decides an active test approval or a requirement applicability request:
 * only the actions the Server requested (Approve, Reject, Not applicable),
 * each with a required reason.
 */
export function ActionDecision({
  auditId,
  review,
  onDecided,
  onRecording,
}: ActionDecisionProps) {
  if (review.subjectKind !== "audit-item-action") return null;
  const intro =
    review.kind === "requirement-applicability" &&
    !review.requestedActions.includes("not_applicable")
      ? undefined
      : ACTION_INTROS[review.kind];
  return (
    <RequestDecision
      key={review.requestId}
      auditId={auditId}
      review={review}
      onDecided={onDecided}
      onRecording={onRecording}
      intro={intro}
    />
  );
}

export interface ReportDecisionProps {
  auditId: string;
  /** The report acceptance request. */
  review: AuditReviewRequest;
  /**
   * The report shown next to the decision. Omitted, the component reads the
   * check's report itself. Either way a pending request is decided only while
   * that report is proposed and carries this request.
   */
  report?: AuditReport | undefined;
  /** Called once the Server recorded the decision and the reads refetched. */
  onDecided?: ((result: AuditActionDecisionResult) => void) | undefined;
  /**
   * True once a decision is sent, false once its refetched request arrived
   * (or it was refused and the reads refreshed). Meanwhile the decision
   * stays, even next to a report that already moved on: keep it mounted.
   */
  onRecording?: ((recording: boolean) => void) | undefined;
}

const REPORT_INTRO =
  "Approving accepts this report and finishes the check. Rejecting ends the check without an accepted report.";

function ReportAcceptance({
  auditId,
  review,
  report,
  onDecided,
  onRecording,
}: ReportDecisionProps) {
  const api = usePublicAPI();
  const pending = review.state === "pending";
  // The same read as the report page, so both share one cached report.
  const read = useQuery({
    queryKey: queryKeys.audits.report(auditId),
    queryFn: () => getAuditReport(api, auditId),
    enabled: report === undefined && pending,
  });
  const current = report ?? read.data;
  // Decided and expired requests need no report.
  let blocked: ReactNode;
  if (pending && current === undefined)
    blocked = read.isError ? (
      <div className="decisions-notice" role="alert">
        <p>
          The report could not be loaded, so this request cannot be decided here
          yet.{" "}
          <button
            type="button"
            className="decisions-text-button"
            onClick={() => void read.refetch()}
          >
            Try again
          </button>
        </p>
        <RequestDetails error={read.error} />
      </div>
    ) : (
      <p className="decisions-quiet" role="status">
        Loading the report…
      </p>
    );
  else if (
    pending &&
    current !== undefined &&
    (current.status !== "proposed" ||
      current.review?.requestId !== review.requestId)
  )
    blocked = (
      <p className="decisions-notice" role="status">
        This report no longer matches its acceptance request. Load the current
        report before deciding.
      </p>
    );
  return (
    <RequestDecision
      auditId={auditId}
      review={review}
      onDecided={onDecided}
      onRecording={onRecording}
      intro={REPORT_INTRO}
      blocked={blocked}
    />
  );
}

/**
 * Accepts or rejects a proposed report. The report is not accepted until the
 * Server records an approval; a request that does not belong to the current
 * proposed report cannot be decided here.
 */
export function ReportDecision(props: ReportDecisionProps) {
  if (props.review.kind !== "report-acceptance") return null;
  return <ReportAcceptance key={props.review.requestId} {...props} />;
}
