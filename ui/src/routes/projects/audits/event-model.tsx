import { Link } from "react-router";
import type { Audit, AuditEvent } from "../../../api/audits";
import {
  checkStateLabel,
  reviewActionLabel,
  reviewKindLabel,
  verdictLabel,
} from "../../../app/vocabulary";
import type { ActivityEntry } from "../../../ui";
import type { CheckLinks } from "./check-links";

/** Durable sequence order wins even if two events share a timestamp. */
export function eventEntry(
  event: AuditEvent,
  links: CheckLinks,
): ActivityEntry {
  const s = event.summary;
  const entry: ActivityEntry = {
    id: `event-${event.sequence}`,
    time: event.createdAt,
    tone: "neutral",
  };
  switch (event.kind) {
    case "audit.created":
      entry.title = "Check created.";
      break;
    case "audit.resumed":
      entry.title = "Check continued.";
      entry.tone = "progress";
      break;
    case "audit.state_changed": {
      const state =
        s.to === undefined
          ? undefined
          : checkStateLabel(s.to as Audit["state"]);
      entry.title =
        state === undefined ? "Check state changed." : `Check: ${state.label}.`;
      entry.tone = state?.tone ?? "neutral";
      break;
    }
    case "round.accepted":
      entry.title =
        s.round === undefined
          ? "Round accepted."
          : s.round === 1
            ? "Check started."
            : "Next round started.";
      entry.text =
        s.items === undefined
          ? undefined
          : `${s.items.toLocaleString("en-US")} items.`;
      entry.tone = "progress";
      break;
    case "round.state_changed":
      entry.title =
        s.to === "closed"
          ? "Round finished."
          : s.to === "assessing"
            ? "Assessing the results."
            : s.to === "executing"
              ? "Round work started."
              : "Round state changed.";
      entry.tone = s.to === "closed" ? "done" : "progress";
      break;
    case "execution.intent_created":
      entry.title = "Work prepared.";
      entry.tone = "progress";
      break;
    case "execution.run_bound":
      entry.title = "Run created.";
      entry.tone = "progress";
      if (s.runId !== undefined)
        entry.text = (
          <Link to={`/runs/${encodeURIComponent(s.runId)}`}>Open run</Link>
        );
      break;
    case "execution.terminal_observed":
      entry.title =
        s.outcome === "failed"
          ? "Run failed."
          : s.outcome === "submission-failed"
            ? "Run could not start."
            : s.outcome === "cancelled"
              ? "Run stopped."
              : s.outcome === "succeeded"
                ? "Run finished."
                : "Run outcome recorded.";
      entry.tone =
        s.outcome === "failed" || s.outcome === "submission-failed"
          ? "blocked"
          : "neutral";
      break;
    case "execution.collected":
      entry.title = "Run result collected.";
      entry.text =
        s.disposition === "accepted-result"
          ? "Its result was accepted."
          : s.disposition === "missing-output"
            ? "It produced no result."
            : s.disposition === "invalid-result"
              ? "Its result was invalid."
              : s.disposition === "execution-failed"
                ? "The run failed."
                : s.disposition === "execution-cancelled"
                  ? "The run was cancelled."
                  : s.disposition === "collection-contract-invalid"
                    ? "Its result could not be read."
                    : undefined;
      entry.tone = s.disposition === "accepted-result" ? "done" : "warning";
      break;
    case "review.requested":
      entry.title = "Waiting for your decision:";
      entry.tone = "review";
      entry.text =
        s.kind === undefined ? (
          <Link to={links.section("reviews")}>Open decisions</Link>
        ) : (
          <Link to={links.section("reviews")}>{reviewKindLabel(s.kind)}</Link>
        );
      break;
    case "review.decided": {
      const verdict =
        s.verdict === undefined ? undefined : verdictLabel(s.verdict);
      entry.title = `Decision recorded${verdict === undefined ? (s.action === undefined ? "." : `: ${reviewActionLabel(s.action).label}.`) : `: ${verdict.label}.`}`;
      entry.tone = verdict?.tone ?? "done";
      break;
    }
    case "review.expired":
      entry.title = "Decision request expired.";
      entry.tone = "warning";
      break;
    case "finding.assessed":
      entry.title = "Possible issue assessed.";
      entry.tone = "review";
      break;
    case "finding.proposal_rejected":
      entry.title = "Possible issue proposal was not accepted.";
      entry.tone = "warning";
      break;
    case "audit.report_committed":
      entry.title = "Report is ready.";
      entry.tone = "done";
      entry.text = <Link to={links.section("report")}>Open the report</Link>;
      break;
    case "items.dispatch_closed":
      entry.title = "New item work stopped.";
      break;
    case "audit.dispatch_hold_released":
      entry.title = "Check resources released.";
      break;
    default:
      entry.title = "Check activity recorded.";
      entry.text = event.kind;
  }
  return entry;
}
