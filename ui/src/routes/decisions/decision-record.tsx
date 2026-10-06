import { RecordedTime } from "../../app/recorded-time";
import { IdChip, StatusChip } from "../../ui";
import { DecisionMarkdown } from "./markdown";
import { decisionOutcome, type AuditReviewDecision } from "./model";

import "./decisions.css";

/**
 * One recorded decision: the outcome in the shared vocabulary ("Confirmed ·
 * High", "Not an issue", "Approved"), who recorded it and when, the original
 * of a duplicate, and the reason as written.
 */
export function DecisionRecord({
  decision,
}: {
  decision: AuditReviewDecision;
}) {
  const outcome = decisionOutcome(decision);
  return (
    <div className="decisions-record">
      <p className="decisions-record-head">
        <StatusChip tone={outcome.tone} size="sm">
          {outcome.label}
        </StatusChip>
        <span className="decisions-record-meta">
          by <span className="decisions-record-actor">{decision.actorId}</span>
          {" · "}
          <RecordedTime value={decision.createdAt} />
        </span>
      </p>
      {decision.duplicateTargetId === undefined ? null : (
        <p className="decisions-record-line">
          Duplicate of{" "}
          <IdChip
            value={decision.duplicateTargetId}
            label="ID of the original possible issue"
          />
        </p>
      )}
      <div className="decisions-record-reason">
        <span className="decisions-record-label">Why</span>
        <DecisionMarkdown source={decision.rationale} />
      </div>
    </div>
  );
}
