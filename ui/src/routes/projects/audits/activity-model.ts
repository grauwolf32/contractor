/** Current item activity comes from its retained attempts and results.
 * Whole-check history uses the immutable event API in event-history.tsx. */
import type { Audit, AuditFinding, AuditItem } from "../../../api/audits";
import type { StatusTone } from "../../../app/status-tone";
import { itemNoun, verdictLabel, type ItemKind } from "../../../app/vocabulary";
import type { ActivityEntry } from "../../../ui";
import { firstLine, type CheckEntry } from "./check-model";

type AuditItemAttempt = AuditItem["attempts"][number];

interface TimedEntry extends ActivityEntry {
  /** Sort key, milliseconds. */
  at: number;
}

function at(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function push(
  entries: TimedEntry[],
  id: string,
  time: string | undefined,
  entry: Omit<ActivityEntry, "id" | "time">,
): void {
  const moment = at(time);
  if (moment === undefined || time === undefined) return;
  entries.push({ id, time, at: moment, ...entry });
}

function newestFirst(entries: TimedEntry[]): ActivityEntry[] {
  return entries
    .sort(
      (left, right) => right.at - left.at || left.id.localeCompare(right.id),
    )
    .map((entry): ActivityEntry => ({
      id: entry.id,
      time: entry.time,
      tone: entry.tone,
      title: entry.title,
      text: entry.text,
    }));
}

const OUTCOME_WORDS: Readonly<
  Record<NonNullable<AuditItemAttempt["terminalOutcome"]>, string>
> = {
  succeeded: "finished",
  failed: "failed",
  cancelled: "was cancelled",
  "submission-failed": "could not be started",
};

const DISPOSITIONS: Readonly<
  Record<
    NonNullable<AuditItemAttempt["collectionDisposition"]>,
    { text: string; tone: StatusTone }
  >
> = {
  "accepted-result": { text: "Its result was accepted.", tone: "done" },
  "missing-output": { text: "It produced no result.", tone: "warning" },
  "invalid-result": { text: "Its result was invalid.", tone: "warning" },
  "execution-failed": { text: "The run failed.", tone: "blocked" },
  "execution-cancelled": { text: "The run was cancelled.", tone: "neutral" },
  "collection-contract-invalid": {
    text: "Its result could not be read.",
    tone: "warning",
  },
};

/** "finished", "failed", … for an attempt's technical outcome. */
export function outcomeWords(
  outcome: AuditItemAttempt["terminalOutcome"],
): string {
  return outcome !== undefined && Object.hasOwn(OUTCOME_WORDS, outcome)
    ? OUTCOME_WORDS[outcome]
    : "finished";
}

/** A sentence and tone for how an attempt's result was collected. */
export function dispositionOf(attempt: AuditItemAttempt): {
  text: string | undefined;
  tone: StatusTone;
} {
  const disposition = attempt.collectionDisposition;
  if (disposition !== undefined && Object.hasOwn(DISPOSITIONS, disposition))
    return DISPOSITIONS[disposition];
  if (
    attempt.terminalOutcome === "failed" ||
    attempt.terminalOutcome === "submission-failed"
  )
    return { text: undefined, tone: "blocked" };
  return { text: undefined, tone: "neutral" };
}

function findingEvents(
  entries: TimedEntry[],
  finding: AuditFinding,
  prefix: string,
): void {
  const title = finding.firstProposal.document.title;
  push(entries, `${prefix}finding-${finding.findingId}`, finding.createdAt, {
    tone: "review",
    title: "Proposed a possible issue:",
    text: title,
  });
  const decision = finding.analystDecision;
  if (decision !== undefined) {
    const outcome = verdictLabel(decision.verdict);
    push(
      entries,
      `${prefix}decision-${decision.decisionId}`,
      decision.createdAt,
      {
        tone: outcome.tone,
        title: `Decision recorded: ${outcome.label}.`,
        text: title,
      },
    );
  }
}

/** What happened to one item, newest first. */
export function itemActivity(entry: CheckEntry): ActivityEntry[] {
  const entries: TimedEntry[] = [];
  const { row, item } = entry;
  if (item !== undefined) {
    push(entries, "item", item.createdAt, {
      tone: "neutral",
      title: "Added to the check.",
    });
    for (const attempt of item.attempts) {
      const n = attempt.itemAttempt;
      push(entries, `attempt-${attempt.executionItemId}`, attempt.createdAt, {
        tone: "progress",
        title:
          n > 1
            ? `Retried automatically: attempt ${n} started.`
            : `Attempt ${n} started.`,
      });
      if (attempt.collectedAt !== undefined) {
        const disposition = dispositionOf(attempt);
        push(
          entries,
          `collected-${attempt.executionItemId}`,
          attempt.collectedAt,
          {
            tone: disposition.tone,
            title: `Attempt ${n} ${outcomeWords(attempt.terminalOutcome)}.`,
            text: disposition.text,
          },
        );
      }
    }
  }
  for (const finding of entry.findings) findingEvents(entries, finding, "");
  if (row.coverage.status !== "not-tested") {
    const summary =
      row.details?.resultSummary === undefined
        ? undefined
        : firstLine(row.details.resultSummary);
    push(entries, "result", row.updatedAt, {
      tone: entry.status.tone,
      title: `Result: ${entry.status.label}.`,
      text: summary === "" ? undefined : summary,
    });
  }
  return newestFirst(entries);
}

/** A sentence about what the check is doing now, or undefined once ended. */
export function nowSentence(
  audit: Audit,
  entries: readonly CheckEntry[],
  kind: ItemKind,
): string | undefined {
  const noun = (count: number) => itemNoun(kind, count);
  const inProgress = entries.filter(
    (entry) => entry.status.tone === "progress",
  ).length;
  switch (audit.state) {
    case "draft":
      return `Draft: start the check to create its ${noun(2)}.`;
    case "active":
      return inProgress > 0
        ? `Running: ${inProgress.toLocaleString("en-US")} ${noun(inProgress)} ${inProgress === 1 ? "is" : "are"} being checked now.`
        : "Running.";
    case "waiting_review":
      return "Waiting for your decisions before it can go on.";
    case "paused":
      return audit.outstandingRunCount > 0
        ? "Paused: no new work starts; running work can finish."
        : "Paused: no new work starts.";
    case "finalizing":
      return "Finishing: the results are being put together.";
    case "cancelling":
      return "Stopping: running work is being cancelled.";
    case "deleting":
      return "Being deleted.";
    default:
      return undefined;
  }
}
