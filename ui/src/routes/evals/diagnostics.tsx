import type { EvalExperiment } from "../../api/evals";

const MESSAGES: Record<string, string> = {
  eval_pin_mismatch:
    "The selected versions, inputs or required-equal settings do not match. Review the exact versions and equality policy before preparing again.",
  eval_invalid:
    "Some setup values are invalid. Review the variants, cases and assessment checks, then save the corrected draft.",
  eval_not_ready:
    "The experiment is not ready for this action. Review its setup and required inputs.",
  eval_preparation_unavailable:
    "Preparation is temporarily unavailable. The server will retry; you can safely leave this page.",
  eval_evidence_unavailable:
    "Required evidence is no longer available. Open the affected execution to inspect its artifacts.",
  eval_budget_exhausted:
    "The experiment reached its allowance. Duplicate it to run again with different limits.",
};

export function EvalDiagnostics({
  experiment,
}: {
  experiment: EvalExperiment;
}) {
  if (!experiment.diagnostics?.length) return null;
  return (
    <section className="notice-error eval-diagnostics" role="alert">
      <strong>
        {experiment.state === "draft" || experiment.state === "preparing"
          ? "Preparation needs attention"
          : "Experiment needs attention"}
      </strong>
      {experiment.diagnostics.map((diagnostic, index) => (
        <div key={index}>
          <p>
            {MESSAGES[diagnostic.code] ??
              `The server could not complete this action. Suggested recovery: ${diagnostic.recovery.replaceAll("_", " ")}.`}
          </p>
          {diagnostic.field ? <p>Check: {diagnostic.field}</p> : null}
          <details className="error-details">
            <summary>Diagnostic details</summary>
            <code>{diagnostic.code}</code>
          </details>
        </div>
      ))}
    </section>
  );
}
