/**
 * Activity logs of a check and of one item, built from the timestamps the
 * Server keeps on the check, its items, attempts, possible issues and review
 * requests. There is no event stream yet (S19:1858-1860), so a log is as
 * complete as those records.
 */
import type {
  Audit,
  AuditFinding,
  AuditItem,
  AuditReviewRequest,
} from "../../../api/audits";
import type { StatusTone } from "../../../app/status-tone";
import {
  itemNoun,
  reviewKindLabel,
  verdictLabel,
  type ItemKind,
} from "../../../app/vocabulary";
import type { ActivityEntry } from "../../../ui";
import { entryName, firstLine, type CheckEntry } from "./check-model";
import { describeStopReason } from "./stop-reason";

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

function attemptFailed(attempt: AuditItemAttempt): boolean {
  return (
    attempt.terminalOutcome === "failed" ||
    attempt.terminalOutcome === "submission-failed" ||
    attempt.collectionDisposition === "execution-failed"
  );
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

const END_EVENTS: Partial<
  Record<Audit["state"], { title: string; tone: StatusTone }>
> = {
  completed: { title: "Check finished.", tone: "done" },
  cancelled: { title: "Check stopped.", tone: "neutral" },
  failed: { title: "Check failed.", tone: "blocked" },
};

const NOW_TONES: Partial<Record<Audit["state"], StatusTone>> = {
  draft: "idle",
  waiting_review: "review",
  paused: "warning",
  cancelling: "warning",
  deleting: "neutral",
};

/** The most entries the whole-check log shows. */
export const CHECK_ACTIVITY_LIMIT = 30;

/**
 * What happened in the whole check, newest first: its lifecycle, results of
 * its items, failed attempts, possible issues, decisions and the decisions
 * waiting for the user. `total` counts every event before the limit.
 */
export function checkActivity({
  audit,
  entries,
  findings,
  waiting,
  now,
}: {
  audit: Audit;
  entries: readonly CheckEntry[];
  findings: readonly AuditFinding[];
  waiting: readonly AuditReviewRequest[];
  /** What the check is doing now, shown first. */
  now: string | undefined;
}): { entries: ActivityEntry[]; total: number } {
  const events: TimedEntry[] = [];
  push(events, "created", audit.createdAt, {
    tone: "neutral",
    title: "Check created.",
  });
  push(events, "started", audit.startedAt, {
    tone: "progress",
    title: "Check started.",
  });
  if (audit.state === "paused")
    push(events, "paused", audit.pausedAt, {
      tone: "warning",
      title: "Paused.",
    });
  const end = END_EVENTS[audit.state];
  if (end !== undefined)
    push(events, "finished", audit.finishedAt, {
      ...end,
      text: describeStopReason(audit)?.sentence,
    });
  push(events, "deletion", audit.deletionRequestedAt, {
    tone: "neutral",
    title: "Deletion requested.",
  });
  for (const entry of entries) {
    const name = entryName(entry);
    if (entry.row.coverage.status !== "not-tested")
      push(events, `result-${entry.row.itemId}`, entry.row.updatedAt, {
        tone: entry.status.tone,
        title: `${entry.status.label}:`,
        text: name,
      });
    for (const attempt of entry.item?.attempts ?? []) {
      if (!attemptFailed(attempt)) continue;
      push(
        events,
        `failed-${attempt.executionItemId}`,
        attempt.collectedAt ?? attempt.createdAt,
        {
          tone: "blocked",
          title: `Attempt ${attempt.itemAttempt} ${outcomeWords(attempt.terminalOutcome)}:`,
          text: name,
        },
      );
    }
  }
  for (const finding of findings) findingEvents(events, finding, "check-");
  for (const review of waiting)
    push(events, `waiting-${review.requestId}`, review.createdAt, {
      tone: "review",
      title: "Waiting for your decision:",
      text: reviewKindLabel(review.kind),
    });
  const sorted = newestFirst(events);
  const shown = sorted.slice(0, CHECK_ACTIVITY_LIMIT);
  return {
    entries:
      now === undefined
        ? shown
        : [
            {
              id: "now",
              time: "now",
              tone: NOW_TONES[audit.state] ?? "progress",
              title: now,
            },
            ...shown,
          ],
    total: sorted.length,
  };
}
