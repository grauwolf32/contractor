import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";

import { collectAuditPages } from "../../api/audit-collections";
import {
  createAuditFindingReview,
  decideAuditFinding,
  listAuditReviews,
  type AuditAnalystVerdict,
  type AuditFinding,
  type AuditFindingSeverity,
  type AuditReviewRequest,
} from "../../api/audits";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { RecordedTime } from "../../app/recorded-time";
import { findingStateLabel, severityLabel } from "../../app/vocabulary";
import { MutationDraftKeyring } from "../../mutations/idempotency";
import {
  DecisionBar,
  IdChip,
  Kbd,
  StatusChip,
  useShortcuts,
  type DecisionOption,
  type DecisionSeverity,
} from "../../ui";
import { useAnnouncement } from "./announcement";
import { DecisionRecord } from "./decision-record";
import { DuplicatePicker } from "./duplicate-picker";
import { DecisionIcon } from "./icons";
import { DecisionMarkdown } from "./markdown";
import {
  decisionErrorMessage,
  decisionOutcome,
  findingDecisionBody,
  findingDraftProblem,
  isSeverity,
  latestDecision,
  MAX_RATIONALE_BYTES,
  pickPendingReview,
  RATIONALE_HELP,
  reviewExpired,
  SEVERITY_OPTIONS,
  VERDICT_ACTIONS,
  verdictForState,
  type AuditFindingDecisionResult,
  type FindingDecisionDraft,
  type PendingReviewStatus,
} from "./model";
import { MoreDecisions, type MoreDecision } from "./more-menu";
import { refreshAfterDecision } from "./refresh";
import { RequestDetails } from "./request-details";
import { aiSummary } from "./text";

import "./decisions.css";

/** The next item to decide, e.g. in the Inbox; bound to J. */
export interface FindingDecisionNext {
  label: string;
  onNext: () => void;
}

export interface FindingDecisionProps {
  auditId: string;
  finding: AuditFinding;
  /** Called once the Server recorded the decision and the reads refetched. */
  onDecided?: ((result: AuditFindingDecisionResult) => void) | undefined;
  /** A quiet "next" button; J moves on when shortcuts are on. */
  next?: FindingDecisionNext | undefined;
  /** Moves focus to the first decision control when it appears. */
  autoFocus?: boolean | undefined;
  /**
   * The finding's open review request when the caller already read the
   * check's pending reviews; null when it has none. Omitted, the component
   * reads the finding's pending reviews itself.
   */
  pendingReview?: AuditReviewRequest | null | undefined;
  /**
   * C, R, E and J. Default true; turn off where several decisions share a
   * page, so one key cannot reach another possible issue.
   */
  shortcuts?: boolean | undefined;
}

type FocusTarget = "start" | "reason" | "duplicate" | "root";

interface FocusRequest {
  target: FocusTarget;
  /** Increases with every request, so a repeated target runs again. */
  id: number;
}

const VERDICTS: readonly AuditAnalystVerdict[] = [
  "true_positive",
  "false_positive",
  "needs_evidence",
  "duplicate",
  "reopen",
];

function isVerdict(value: string): value is AuditAnalystVerdict {
  return (VERDICTS as readonly string[]).includes(value);
}

const REJECTION_REASONS: Readonly<
  Record<NonNullable<AuditFinding["rejectionReason"]>, string>
> = {
  "false-positive": "False positive",
  policy: "Policy",
  "out-of-scope": "Out of scope",
};

/**
 * The open request a decision can use: an expired one cannot be decided, and
 * opening a review for the current revision replaces it.
 */
function decidableReview(
  review: AuditReviewRequest | undefined,
): AuditReviewRequest | undefined {
  return review !== undefined && !reviewExpired(review, Date.now())
    ? review
    : undefined;
}

interface Submission {
  draft: FindingDecisionDraft;
  /** The open request to decide, when one is known. */
  review: AuditReviewRequest | undefined;
  /** The finding revision the user saw. */
  findingRevision: number;
}

/**
 * The reason behind a decided state that the finding does not carry itself
 * (duplicate, needs evidence): read from the finding's decided reviews on
 * request, so lists of possible issues make no extra reads.
 */
function StateReason({
  auditId,
  finding,
}: {
  auditId: string;
  finding: AuditFinding;
}) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const history = useQuery({
    queryKey: [
      ...queryKeys.audits.reviews(auditId, finding.findingId),
      "decided",
    ],
    queryFn: () =>
      collectAuditPages((cursor) =>
        listAuditReviews(api, auditId, {
          finding: finding.findingId,
          state: "decided",
          ...(cursor === undefined ? {} : { cursor }),
        }),
      ),
    enabled: open,
  });
  const decision =
    history.data === undefined
      ? undefined
      : latestDecision(history.data.items, verdictForState(finding.state));
  return (
    <div className="decisions-state-reason">
      <button
        type="button"
        className="decisions-text-button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? "Hide the reason" : "Show the reason"}
      </button>
      <div id={panelId} hidden={!open}>
        {!open ? null : history.isPending ? (
          <p className="decisions-quiet">Loading the decision…</p>
        ) : history.isError ? (
          <p className="decisions-quiet" role="alert">
            The decision could not be loaded.{" "}
            <button
              type="button"
              className="decisions-text-button"
              onClick={() => void history.refetch()}
            >
              Try again
            </button>
          </p>
        ) : decision === undefined ? (
          <p className="decisions-quiet">
            No recorded decision explains this state.
          </p>
        ) : (
          <>
            <p className="decisions-record-meta">
              by{" "}
              <span className="decisions-record-actor">{decision.actorId}</span>
              {" · "}
              <RecordedTime value={decision.createdAt} />
            </p>
            <div className="decisions-record-reason">
              <span className="decisions-record-label">Why</span>
              <DecisionMarkdown source={decision.rationale} />
            </div>
            {history.data.truncated ? (
              <p className="decisions-quiet">
                This possible issue has more decisions than shown here; a newer
                one may be missing.
              </p>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

/** The decision a decided possible issue carries now. */
function CurrentOutcome({
  auditId,
  finding,
}: {
  auditId: string;
  finding: AuditFinding;
}) {
  if (finding.analystDecision !== undefined)
    return <DecisionRecord decision={finding.analystDecision} />;
  const state = findingStateLabel(finding.state);
  return (
    <div className="decisions-record">
      <p className="decisions-record-head">
        <StatusChip tone={state.tone} size="sm">
          {state.label}
        </StatusChip>
        {finding.rejectionReason === undefined ? null : (
          <span className="decisions-record-meta">
            Reason:{" "}
            {REJECTION_REASONS[finding.rejectionReason] ??
              finding.rejectionReason}
          </span>
        )}
      </p>
      {finding.duplicateTargetId === undefined ? null : (
        <p className="decisions-record-line">
          Duplicate of{" "}
          <IdChip
            value={finding.duplicateTargetId}
            label="ID of the original possible issue"
          />
        </p>
      )}
      <StateReason auditId={auditId} finding={finding} />
    </div>
  );
}

function FindingDecisionPanel({
  auditId,
  finding,
  onDecided,
  next,
  autoFocus = false,
  pendingReview,
  shortcuts = true,
}: FindingDecisionProps) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const findingId = finding.findingId;
  const searchId = useId();
  const root = useRef<HTMLDivElement>(null);
  // Where focus goes after the render that shows the target.
  const [focusRequest, setFocusRequest] = useState<FocusRequest | null>(
    autoFocus ? { target: "start", id: 1 } : null,
  );
  const handledFocus = useRef(0);

  // An open review request is reused; without one, recording opens it.
  const lookup = pendingReview === undefined;
  const reviews = useQuery({
    queryKey: [...queryKeys.audits.reviews(auditId, findingId), "pending"],
    queryFn: () =>
      listAuditReviews(api, auditId, { finding: findingId, state: "pending" }),
    enabled: lookup,
  });
  const status: PendingReviewStatus = !lookup
    ? { review: pendingReview ?? undefined, stale: false }
    : reviews.data === undefined
      ? { stale: false }
      : pickPendingReview(reviews.data.items, finding);
  const statusLoading =
    lookup && reviews.data === undefined && reviews.isPending;
  const statusError =
    lookup && reviews.data === undefined ? reviews.error : null;

  const [verdict, setVerdict] = useState<AuditAnalystVerdict | undefined>();
  const [chosenSeverity, setChosenSeverity] = useState<
    AuditFindingSeverity | undefined
  >();
  const [rationale, setRationale] = useState("");
  const [duplicateTarget, setDuplicateTarget] = useState("");
  const [changing, setChanging] = useState(false);
  const [problem, setProblem] = useState<string | undefined>();
  // The finding revision a recorded decision was made on: until the
  // refetched finding replaces it, the bar stays closed.
  const [recordedRevision, setRecordedRevision] = useState<number | null>(null);
  // "Decision recorded: …" outlives that gate, so screen readers read it.
  const announcement = useAnnouncement();
  const [keyrings] = useState(() => ({
    create: new MutationDraftKeyring<Record<string, string | number>>(
      "audit-finding-review",
    ),
    decide: new MutationDraftKeyring<Record<string, string | number>>(
      "audit-finding-review",
    ),
  }));
  // The analyst's own earlier rating stays chosen; the AI's suggestion never is.
  const severity = chosenSeverity ?? finding.analystSeverity;

  function requestFocus(target: FocusTarget) {
    setFocusRequest((previous) => ({ target, id: (previous?.id ?? 0) + 1 }));
  }

  // A focus request runs after the render that shows its target. A target
  // that is still disabled (the review status is loading) waits for a later
  // render.
  useEffect(() => {
    const container = root.current;
    if (
      focusRequest === null ||
      focusRequest.id === handledFocus.current ||
      container === null
    )
      return;
    const target = focusRequest.target;
    let element: HTMLElement | null = null;
    if (target === "start")
      element = container.querySelector<HTMLElement>(
        "[data-decisions-start], button[aria-pressed]",
      );
    else if (target === "reason")
      element = container.querySelector<HTMLElement>("textarea");
    else if (target === "duplicate")
      element = document.getElementById(searchId);
    else if (
      document.activeElement === null ||
      document.activeElement === document.body ||
      container.contains(document.activeElement)
    )
      element = container;
    if (element?.matches(":disabled")) return;
    handledFocus.current = focusRequest.id;
    element?.focus();
  });

  const decide = useMutation({
    mutationFn: async ({ draft, review, findingRevision }: Submission) => {
      let request = decidableReview(review);
      if (request === undefined) {
        const created = await createAuditFindingReview(api, {
          auditId,
          findingId,
          expectedRevision: findingRevision,
          idempotencyKey: keyrings.create.keyFor({
            operation: "create",
            auditId,
            findingId,
            revision: findingRevision,
          }),
        });
        if (
          created.state !== "pending" ||
          created.kind !== "finding-triage" ||
          (created.findingId ?? created.subjectId) !== findingId ||
          created.subjectRevision !== findingRevision
        )
          throw new Error(
            "the Server opened a review for a different subject. Nothing was decided.",
          );
        request = created;
      }
      if (!request.requestedActions.includes(draft.verdict))
        throw new Error(
          `this review does not offer “${VERDICT_ACTIONS[draft.verdict].replace("…", "")}”. Nothing was decided.`,
        );
      const decision = findingDecisionBody(draft);
      return decideAuditFinding(api, {
        auditId,
        requestId: request.requestId,
        expectedRevision: request.revision,
        idempotencyKey: keyrings.decide.keyFor({
          operation: "decide",
          auditId,
          requestId: request.requestId,
          revision: request.revision,
          ...decision,
        }),
        decision,
      });
    },
    onSuccess: async (result, { findingRevision }) => {
      await refreshAfterDecision(queryClient, auditId);
      setRecordedRevision(findingRevision);
      announcement.announce(
        `Decision recorded: ${decisionOutcome(result.decision).label}.`,
      );
      setVerdict(undefined);
      setChosenSeverity(undefined);
      setRationale("");
      setDuplicateTarget("");
      setChanging(false);
      setProblem(undefined);
      requestFocus("root");
      onDecided?.(result);
    },
    onError: () => refreshAfterDecision(queryClient, auditId),
  });

  const decided = finding.state !== "proposed";
  // Until the refetched finding arrives, only the announcement shows.
  const awaitingRefresh =
    recordedRevision !== null && recordedRevision === finding.revision;
  const open =
    !awaitingRefresh && (!decided || changing || status.review !== undefined);

  useShortcuts(
    { j: () => next?.onNext() },
    {
      enabled: shortcuts && next !== undefined && !open && !awaitingRefresh,
    },
  );

  function offered(candidate: AuditAnalystVerdict): boolean {
    return (
      status.review === undefined ||
      status.review.requestedActions.includes(candidate)
    );
  }

  // Every edit of the draft: the last answer's announcement and a client-side
  // problem no longer apply.
  function edited() {
    setProblem(undefined);
    announcement.clear();
  }

  function choose(candidate: AuditAnalystVerdict, focus?: FocusTarget) {
    setVerdict(candidate);
    edited();
    if (focus !== undefined) requestFocus(focus);
  }

  function insertSummary() {
    const summary = aiSummary(finding);
    setRationale((current) =>
      current.includes(summary)
        ? current
        : current.trim() === ""
          ? summary
          : `${current.trimEnd()}\n\n${summary}`,
    );
    edited();
    requestFocus("reason");
  }

  function submit() {
    if (verdict === undefined) return;
    const draft: FindingDecisionDraft = {
      verdict,
      severity: verdict === "true_positive" ? severity : undefined,
      duplicateTargetId: verdict === "duplicate" ? duplicateTarget : undefined,
      rationale,
    };
    const issue = findingDraftProblem(draft, findingId);
    if (issue !== undefined) {
      setProblem(issue);
      if (verdict === "duplicate") requestFocus("duplicate");
      return;
    }
    setProblem(undefined);
    decide.mutate({
      draft,
      review: status.review,
      findingRevision: finding.revision,
    });
  }

  function startChange() {
    setChanging(true);
    announcement.clear();
    requestFocus("start");
  }

  // Discards the draft; the current decision stays as it is.
  function keepCurrent() {
    setChanging(false);
    setVerdict(undefined);
    setChosenSeverity(undefined);
    setRationale("");
    setDuplicateTarget("");
    edited();
    decide.reset();
    requestFocus("start");
  }

  const key = (letter: string) => (shortcuts ? letter : undefined);
  const options: DecisionOption[] = [];
  if (offered("true_positive"))
    options.push({
      id: "true_positive",
      label: VERDICT_ACTIONS.true_positive,
      shortcut: key("c"),
      tone: "primary",
      icon: <DecisionIcon name="confirm" />,
    });
  if (offered("false_positive"))
    options.push({
      id: "false_positive",
      label: VERDICT_ACTIONS.false_positive,
      shortcut: key("r"),
      icon: <DecisionIcon name="reject" />,
    });
  if (offered("needs_evidence"))
    options.push({
      id: "needs_evidence",
      label: VERDICT_ACTIONS.needs_evidence,
      shortcut: key("e"),
      icon: <DecisionIcon name="unsure" />,
    });
  // Duplicate and Reopen live in More; the chosen one shows as pressed.
  if (verdict === "duplicate")
    options.push({ id: "duplicate", label: "Duplicate" });
  if (verdict === "reopen") options.push({ id: "reopen", label: "Reopen" });

  const more: MoreDecision[] = [];
  if (offered("duplicate"))
    more.push({
      id: "duplicate",
      label: VERDICT_ACTIONS.duplicate,
      onSelect: () => choose("duplicate", "duplicate"),
    });
  // Reopen returns a decided possible issue to review.
  if (decided && offered("reopen"))
    more.push({
      id: "reopen",
      label: VERDICT_ACTIONS.reopen,
      onSelect: () => choose("reopen", "reason"),
    });

  const suggestion = finding.firstProposal.document.severity_suggestion;
  const severityField: DecisionSeverity | undefined = offered("true_positive")
    ? {
        options: SEVERITY_OPTIONS,
        value: severity,
        onChange: (value) => {
          if (!isSeverity(value)) return;
          setChosenSeverity(value);
          announcement.clear();
        },
        required: verdict === "true_positive",
        hint:
          severity === undefined
            ? suggestion === ""
              ? "Not set yet. Pick one when you confirm."
              : `Not set yet. The AI suggests ${severityLabel(suggestion)}.`
            : verdict !== undefined && verdict !== "true_positive"
              ? "Saved only when you confirm."
              : undefined,
      }
    : undefined;

  const disabledReason = statusLoading
    ? "Checking for an open review…"
    : statusError !== null
      ? "The review status could not be loaded."
      : status.stale
        ? "This possible issue changed since it was loaded."
        : undefined;
  const decideMessage =
    decide.error === null
      ? undefined
      : decisionErrorMessage(decide.error, "finding");
  const error =
    statusError !== null ? (
      <>
        {statusError.message}{" "}
        <button
          type="button"
          className="decisions-text-button"
          onClick={() => void reviews.refetch()}
        >
          Try again
        </button>
      </>
    ) : status.stale ? (
      <>
        Load the latest version before deciding.{" "}
        <button
          type="button"
          className="decisions-text-button"
          onClick={() => void refreshAfterDecision(queryClient, auditId)}
        >
          Load the latest version
        </button>
      </>
    ) : (
      (problem ?? decideMessage)
    );
  // The Public API error behind the bar's message, for "Request details".
  const barError: { error: unknown; text: string | undefined } | null =
    statusError !== null
      ? { error: statusError, text: statusError.message }
      : status.stale || problem !== undefined || decide.error === null
        ? null
        : { error: decide.error, text: decideMessage };

  const canKeep = changing && status.review === undefined;

  return (
    <div
      ref={root}
      className="decisions-finding"
      role="group"
      aria-label={`Decision on ${finding.firstProposal.document.title}`}
      tabIndex={-1}
    >
      <p className="decisions-status" role="status">
        {announcement.text}
      </p>
      {!open && !awaitingRefresh && decide.error !== null ? (
        // The refetch after a refused decision closed the bar (someone else
        // decided); the explanation stays.
        <div className="decisions-notice decisions-inset" role="alert">
          <p>{decideMessage}</p>
          <RequestDetails error={decide.error} explanation={decideMessage} />
        </div>
      ) : null}
      {awaitingRefresh ? null : (
        <>
          {decided ? (
            <section
              className="decisions-current"
              aria-label="Current decision"
            >
              <CurrentOutcome auditId={auditId} finding={finding} />
              {open && !canKeep ? null : (
                <div className="decisions-current-actions">
                  {canKeep ? (
                    <button
                      type="button"
                      className="ui-btn"
                      data-size="sm"
                      data-variant="ghost"
                      disabled={decide.isPending}
                      onClick={keepCurrent}
                    >
                      Keep current decision
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="ui-btn"
                      data-size="sm"
                      data-decisions-start=""
                      onClick={startChange}
                    >
                      Change decision
                    </button>
                  )}
                  {next === undefined || open ? null : (
                    <button
                      type="button"
                      className="ui-btn"
                      data-size="sm"
                      data-variant="ghost"
                      aria-keyshortcuts={shortcuts ? "J" : undefined}
                      onClick={next.onNext}
                    >
                      <DecisionIcon name="next" />
                      <span>{next.label}</span>
                      {shortcuts ? (
                        <span className="ui-kbd-hint" aria-hidden="true">
                          <Kbd>J</Kbd>
                        </span>
                      ) : null}
                    </button>
                  )}
                </div>
              )}
            </section>
          ) : null}
          {open ? (
            <DecisionBar
              options={options}
              selected={verdict}
              onSelect={(id) => {
                if (isVerdict(id)) choose(id);
              }}
              severity={severityField}
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
              error={error}
              disabledReason={disabledReason}
              more={
                more.length === 0 ? undefined : <MoreDecisions items={more} />
              }
              next={
                next === undefined
                  ? undefined
                  : {
                      label: next.label,
                      shortcut: key("j"),
                      onNext: next.onNext,
                    }
              }
              extra={
                <>
                  {verdict === "duplicate" ? (
                    <DuplicatePicker
                      auditId={auditId}
                      findingId={findingId}
                      value={duplicateTarget}
                      onChange={(target) => {
                        setDuplicateTarget(target);
                        edited();
                      }}
                      searchId={searchId}
                    />
                  ) : null}
                  <div className="decisions-assist">
                    <button
                      type="button"
                      className="ui-btn"
                      data-size="xs"
                      data-variant="ghost"
                      onClick={insertSummary}
                    >
                      Use AI summary
                    </button>
                  </div>
                </>
              }
            />
          ) : null}
          {open && barError !== null ? (
            // DecisionBar's error slot is a paragraph; the disclosure follows.
            <RequestDetails
              error={barError.error}
              explanation={barError.text}
              className="decisions-under-bar"
            />
          ) : null}
        </>
      )}
    </div>
  );
}

/**
 * Decides a possible issue in place: Confirm issue (with severity), Not an
 * issue, Needs evidence, and Duplicate… or Reopen under More, each with a
 * required reason. Recording reuses the finding's open review request or
 * opens one for its current revision, then records the decision; both
 * requests carry the expected revision and an idempotency key. A refused
 * decision refetches and explains; nothing is retried or shown as decided
 * before the Server answers.
 *
 * A decided possible issue shows its current decision with "Change decision".
 */
export function FindingDecision(props: FindingDecisionProps) {
  // A different possible issue starts with an empty form.
  return (
    <FindingDecisionPanel
      key={`${props.auditId}\u0000${props.finding.findingId}`}
      {...props}
    />
  );
}
