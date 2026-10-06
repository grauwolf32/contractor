import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import {
  decideAuditAction,
  type AuditReport,
  type AuditReviewAction,
  type AuditReviewRequest,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import { DecisionBar, type DecisionOption } from "../../ui";
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
import { refreshAfterDecision } from "./refresh";

import "./decisions.css";

function isAction(value: string): value is AuditReviewAction {
  return REVIEW_ACTIONS.some((candidate) => candidate.action === value);
}

interface RequestDecisionProps {
  auditId: string;
  review: AuditReviewRequest;
  onDecided?: ((result: AuditActionDecisionResult) => void) | undefined;
  /** What the actions do, shown under them. */
  intro?: ReactNode;
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
  intro,
}: RequestDecisionProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [action, setAction] = useState<AuditReviewAction | undefined>();
  const [rationale, setRationale] = useState("");
  const [problem, setProblem] = useState<string | undefined>();
  const [recorded, setRecorded] = useState<{
    revision: number;
    label: string;
  } | null>(null);
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
      setRecorded({
        revision: choice.revision,
        label: decisionOutcome(result.decision).label,
      });
      setAction(undefined);
      setRationale("");
      onDecided?.(result);
    },
    onError: () => refreshAfterDecision(queryClient, auditId),
  });

  // Until the refetched request arrives, say what the Server recorded.
  const awaitingRefresh =
    recorded !== null && recorded.revision === review.revision;
  const options: DecisionOption[] = REVIEW_ACTIONS.filter((candidate) =>
    review.requestedActions.includes(candidate.action),
  ).map((candidate) => ({
    id: candidate.action,
    label: candidate.label,
    tone: candidate.action === "approve" ? "primary" : "secondary",
  }));

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
  if (awaitingRefresh) body = null;
  else if (review.state === "decided")
    body =
      review.decision === undefined ? (
        <p className="decisions-quiet">Decided.</p>
      ) : (
        <DecisionRecord decision={review.decision} />
      );
  else if (review.state === "expired")
    body = (
      <p className="decisions-quiet">
        This request expired without a decision.
      </p>
    );
  else if (options.length === 0)
    body = (
      <p className="decisions-quiet">
        This request offers no decision that can be made here.
      </p>
    );
  else
    body = (
      <DecisionBar
        options={options}
        selected={action}
        onSelect={(id) => {
          if (isAction(id)) {
            setAction(id);
            setProblem(undefined);
          }
        }}
        rationale={{
          value: rationale,
          onChange: (value) => {
            setRationale(value);
            setProblem(undefined);
          },
          placeholder: RATIONALE_HELP,
          maxLength: MAX_RATIONALE_BYTES,
        }}
        onSubmit={submit}
        pending={decide.isPending}
        error={
          problem ??
          (decide.error === null
            ? undefined
            : decisionErrorMessage(decide.error, "request"))
        }
        extra={
          intro === undefined ? undefined : (
            <p className="decisions-intro">{intro}</p>
          )
        }
      />
    );

  return (
    <div className="decisions-request">
      <p className="decisions-status" role="status">
        {awaitingRefresh ? `Decision recorded: ${recorded.label}.` : ""}
      </p>
      {review.state !== "pending" && decide.error !== null ? (
        // The refetch after a refused decision shows the request settled
        // elsewhere; the explanation stays.
        <p className="decisions-notice decisions-inset" role="alert">
          {decisionErrorMessage(decide.error, "request")}
        </p>
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
      intro={intro}
    />
  );
}

export interface ReportDecisionProps {
  auditId: string;
  /** The report acceptance request. */
  review: AuditReviewRequest;
  /**
   * The report read with the request. A pending request is only decided
   * next to the proposed report it was opened for.
   */
  report?: AuditReport | undefined;
  /** Called once the Server recorded the decision and the reads refetched. */
  onDecided?: ((result: AuditActionDecisionResult) => void) | undefined;
}

/**
 * Accepts or rejects a proposed report. The report is not accepted until the
 * Server records an approval; a request that does not belong to the report
 * shown cannot be decided here.
 */
export function ReportDecision({
  auditId,
  review,
  report,
  onDecided,
}: ReportDecisionProps) {
  if (review.kind !== "report-acceptance") return null;
  const mismatch =
    report !== undefined &&
    (report.status !== "proposed" ||
      report.review?.requestId !== review.requestId);
  if (review.state === "pending" && mismatch)
    return (
      <p className="decisions-notice" role="status">
        This report no longer matches its acceptance request. Load the current
        report before deciding.
      </p>
    );
  return (
    <RequestDecision
      key={review.requestId}
      auditId={auditId}
      review={review}
      onDecided={onDecided}
      intro="Approving accepts this report and finishes the check. Rejecting ends the check without an accepted report."
    />
  );
}
