import type { RunStatus } from "../../api/runs";
import { plural } from "../../app/format";
import { runStateLabel } from "../../app/vocabulary";
import { useSession } from "../../auth/session";
import { ButtonLink, StatusChip, StatusGlyph, type StatusTone } from "../../ui";
import { recoveryProgress } from "./recovery";
import type { RunTriage } from "./triage";
import { runRecoveryReason } from "../../app/vocabulary";

function triageTitle(run: RunStatus, triage: RunTriage): string {
  const stage = triage.stage === undefined ? "" : ` in ${triage.stage}`;
  switch (run.state) {
    case "failed":
      return `Run failed${stage}`;
    case "succeeded":
      return "Run completed successfully";
    case "cancelled":
      return "Run was cancelled";
    case "cancelling":
      return "Cancellation is in progress";
    case "pending":
      return "Run is queued";
    case "waiting":
      return "Waiting for the model";
    case "initializing":
      return "Run is initializing";
    case "running":
      return triage.stage === undefined
        ? "Run is in progress"
        : `Running ${triage.stage}`;
  }
}

function triageDescription(run: RunStatus, triage: RunTriage): string {
  if (triage.issue !== undefined) {
    return triage.issue.message;
  }
  switch (run.state) {
    case "failed":
      return "No failure reason was reported. Inspect the latest attempt for details.";
    case "succeeded":
      return triage.outputCount === 0
        ? "All work completed. This Run has no outputs."
        : `${plural(triage.outputCount, "output is", "outputs are")} ready to inspect.`;
    case "cancelled":
      return "Work stopped and cleanup completed.";
    case "cancelling":
      return "Stopping active work and cleaning up.";
    case "pending":
      return "Execution will start when resources and the model are available.";
    case "waiting":
      return "The current invocation is preserved while the model recovers.";
    case "initializing":
      return "Preparing to start the first stage.";
    case "running":
      return "Follow the current stage and its progress under Technical details.";
  }
}

function retryabilityLabel(issue: NonNullable<RunTriage["issue"]>): {
  label: string;
  tone: StatusTone;
} {
  if (issue.source === "cancellation") {
    return { label: "User requested", tone: "neutral" };
  }
  if (issue.retryable === undefined) {
    return { label: "Retryability unknown", tone: "neutral" };
  }
  return issue.retryable
    ? { label: "Retryable", tone: "warning" }
    : { label: "Not retryable", tone: "blocked" };
}

/**
 * Model recovery of a waiting Run, for the triage summary. The Retry model
 * connection action itself sits in the Run page header.
 */
function RecoveryStatus({ run }: { run: RunStatus }) {
  const recovery = run.recovery;
  if (recovery === undefined) return null;
  return (
    <div className="runs-recovery" role="status">
      <StatusGlyph tone={recovery.requiresRetry ? "blocked" : "progress"} />
      <div>
        <strong>{runRecoveryReason(recovery.code)}</strong>
        <p>{recoveryProgress(recovery)}</p>
        <p>
          Completed work is retained. You can cancel this Run while it waits.
        </p>
      </div>
    </div>
  );
}

/**
 * The top of the Run page: what happened (state, primary cause, model
 * recovery), the next step and shortcuts to the focused attempt and the
 * results.
 */
export function RunTriageSummary({
  run,
  triage,
  onOpenAttempt,
}: {
  run: RunStatus;
  triage: RunTriage;
  onOpenAttempt: (stageExecutionId: string) => void;
}) {
  const { session } = useSession();
  const canOperate =
    session?.principal.capabilities.includes("operations") === true;
  const focusedAttempt = triage.stageExecutionId;
  const state = runStateLabel(run.state);
  const issue = triage.issue;
  const retry = issue === undefined ? undefined : retryabilityLabel(issue);
  return (
    <section
      className={`runs-triage run-triage run-triage-${run.state}`}
      aria-labelledby="run-triage-title"
    >
      <div className="runs-triage-head">
        <StatusGlyph tone={state.tone} size={20} />
        <div className="runs-triage-heading">
          <h2 className="runs-h2" id="run-triage-title">
            {triageTitle(run, triage)}
          </h2>
          {issue === undefined ? (
            <p className="runs-triage-text">{triageDescription(run, triage)}</p>
          ) : null}
        </div>
      </div>
      {issue === undefined || retry === undefined ? null : (
        <div className="runs-cause" data-source={issue.source}>
          <p className="runs-label">
            {issue.source === "cancellation"
              ? "Lifecycle reason"
              : "Primary cause"}
          </p>
          <div className="runs-cause-line">
            <code>{issue.code}</code>
            <StatusChip tone={retry.tone} size="sm" glyph={false}>
              {retry.label}
            </StatusChip>
          </div>
          <p className="runs-cause-message">{triageDescription(run, triage)}</p>
          {issue.participant === undefined ? null : (
            <small>
              Reported by {issue.participant}
              {issue.logicalAgent === undefined ? "" : ` ${issue.logicalAgent}`}
            </small>
          )}
        </div>
      )}
      <RecoveryStatus run={run} />
      <div className="runs-next" data-kind={triage.guidance.kind}>
        <p className="runs-label">Next step</p>
        <p className="runs-next-title">{triage.guidance.title}</p>
        <p className="runs-next-text">{triage.guidance.message}</p>
        {run.state !== "failed" ? null : run.resumeStageExecutionId ===
          undefined ? (
          <p className="runs-next-text">
            Continuation is unavailable: cleanup must finish and the Run must
            have a failed stage. Audit-managed and evaluation Runs use their own
            lifecycle controls.
          </p>
        ) : (
          <p className="runs-next-text">
            Continue from failed stage keeps the successful stages and their
            results, and gives the failed stage a new attempt.
          </p>
        )}
        {(run.state === "succeeded" && triage.outputCount > 0) ||
        (triage.guidance.operationsPath !== undefined && canOperate) ? (
          <div className="runs-next-links">
            {run.state === "succeeded" && triage.outputCount > 0 ? (
              <a
                className="ui-btn"
                data-size="sm"
                data-variant="primary"
                href="#run-outputs"
              >
                Read results
              </a>
            ) : null}
            {triage.guidance.operationsPath === undefined ||
            !canOperate ? null : (
              <ButtonLink size="sm" to={triage.guidance.operationsPath}>
                Open Operations diagnostics
              </ButtonLink>
            )}
          </div>
        ) : null}
      </div>
      {focusedAttempt === undefined && triage.outputCount === 0 ? null : (
        <nav className="runs-shortcuts" aria-label="Run triage shortcuts">
          {focusedAttempt === undefined ? null : (
            <a
              href={`#attempt-${focusedAttempt}`}
              onClick={() => onOpenAttempt(focusedAttempt)}
            >
              Inspect focused attempt
            </a>
          )}
          {triage.outputCount === 0 ? null : (
            <a href="#run-outputs">Preview outputs</a>
          )}
        </nav>
      )}
    </section>
  );
}
