import { useId } from "react";
import type { EvalDraft, EvalExperiment } from "../../api/evals";
import {
  IdChip,
  StatusChip,
  TechnicalDetails,
  type StatusTone,
} from "../../ui";
import { ArmKey } from "./common";
import {
  evaluatorLabel,
  executionKindLabel,
  expectedMembersText,
  pinLabel,
} from "./labels";

const ORIGIN_LABELS: Record<string, string> = {
  observed: "Observed",
  "producer-supplied": "Producer supplied",
  unavailable: "Unavailable",
};

const PIN_RESULTS: Record<string, { label: string; tone: StatusTone }> = {
  equal: { label: "Equal", tone: "done" },
  different: { label: "Different", tone: "neutral" },
  unavailable: { label: "Unavailable", tone: "warning" },
};

function pinResult(status: string) {
  return PIN_RESULTS[status] ?? { label: status, tone: "neutral" as const };
}

function dimensions(values: readonly string[]): string {
  return values.map(pinLabel).join(", ") || "None declared";
}

export function EvalReadiness({
  draft,
  experiment,
}: {
  draft?: EvalDraft | undefined;
  experiment?: EvalExperiment | undefined;
}) {
  const heading = useId();
  const setup = draft ?? experiment?.setup;
  if (!setup) return null;
  const expected = draft
    ? draft.caseIds.length * 2 * draft.repetitions
    : experiment?.expectedMembers;
  const arms = (
    [
      ["a", setup.comparison.baseline, "A · Baseline"],
      ["b", setup.comparison.candidate, "B · Candidate"],
    ] as const
  ).map(([arm, id, label]) => ({
    arm,
    label,
    variant: setup.variants.find((variant) => variant.id === id)!,
  }));
  const armOf = (variantId: string) =>
    arms.find((entry) => entry.variant?.id === variantId)?.arm;
  return (
    <section className="eval-section eval-readiness" aria-labelledby={heading}>
      <div className="eval-step-heading">
        <h2 id={heading}>Readiness and setup</h2>
      </div>
      <p className="eval-matrix">
        <strong>
          {expected === undefined
            ? "Expected members unknown"
            : expectedMembersText(expected)}
        </strong>
        {setup.caseIds && setup.repetitions
          ? ` · ${setup.caseIds.length} cases × 2 variants × ${setup.repetitions} repetitions`
          : ""}
      </p>
      <div className="eval-arms">
        {arms.map(({ arm, label, variant: v }) => (
          <section className="eval-panel eval-arm" data-arm={arm} key={arm}>
            <h3>
              <ArmKey arm={arm}>{label}</ArmKey>
            </h3>
            <p className="eval-arm-selector">
              <span>{executionKindLabel(v.kind)}</span>{" "}
              {v.selector ? (
                <IdChip
                  value={v.selector}
                  display={v.selector}
                  label={`${arm.toUpperCase()} version`}
                />
              ) : (
                <span className="eval-muted">Not chosen yet</span>
              )}
            </p>
            <dl className="eval-facts">
              <dt>Runtime labels</dt>
              <dd>{v.runtimeLabels?.join(", ") || "Default runtime"}</dd>
              <dt>Parameters</dt>
              <dd>
                {Object.entries(v.parameters ?? {})
                  .map(([k, val]) => `${k}: ${val}`)
                  .join("; ") || "Case parameters"}
              </dd>
              <dt>Input mapping</dt>
              <dd>
                {Object.entries(v.inputMapping ?? {})
                  .map(([k, val]) => `${k} → ${val}`)
                  .join("; ") || "Same role names"}
              </dd>
              <dt>Output mapping</dt>
              <dd>
                {Object.entries(v.outputMapping ?? {})
                  .map(([k, val]) => `${k} → ${val}`)
                  .join("; ") || "Same role names"}
              </dd>
            </dl>
            {Object.keys(v.executionConfig).length ? (
              <details className="eval-disclosure">
                <summary>Execution overrides</summary>
                <div className="eval-disclosure-body">
                  <pre className="eval-code">
                    {JSON.stringify(v.executionConfig, null, 2)}
                  </pre>
                </div>
              </details>
            ) : null}
          </section>
        ))}
      </div>
      <dl className="eval-facts eval-facts-wide">
        <dt>Dataset revision</dt>
        <dd>
          {setup.dataset
            ? `${setup.dataset.id} · ${setup.dataset.revision}`
            : "External source"}
        </dd>
        <dt>Concurrency</dt>
        <dd>{setup.budgets.maxInFlight} members</dd>
        <dt>Member allowance</dt>
        <dd>{setup.budgets.maxMembers}</dd>
        <dt>Time allowance</dt>
        <dd>{setup.budgets.wallMs / 60000} minutes</dd>
        <dt>Observed token threshold</dt>
        <dd>{setup.budgets.maxObservedTotalTokens ?? "Not set"}</dd>
        <dt>Quality gates</dt>
        <dd>
          Candidate pass ≥ {setup.comparison.gates.minCandidateEndToEndPass};
          quality drop ≤ {setup.comparison.gates.maxQualityDrop}
          {setup.comparison.gates.maxTotalTokensRatio === undefined
            ? ""
            : `; token ratio ≤ ${setup.comparison.gates.maxTotalTokensRatio}`}
        </dd>
        <dt>Required equal</dt>
        <dd>{dimensions(setup.comparison.requiredEqual)}</dd>
        <dt>Allowed differences</dt>
        <dd>{dimensions(setup.comparison.allowedDifferences)}</dd>
      </dl>
      <h3 className="eval-subheading">Criteria</h3>
      {setup.checks.length ? (
        <ul className="eval-criteria" role="list">
          {setup.checks.map((check) => (
            <li key={check.id}>
              <code>{check.id}</code> · {evaluatorLabel(check.evaluator)} ·{" "}
              {check.required ? "required" : "optional"}
              {check.rubricRevision ? ` · rubric ${check.rubricRevision}` : ""}
              {check.parameters
                ? ` · ${Object.entries(check.parameters)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join(", ")}`
                : ""}
            </li>
          ))}
        </ul>
      ) : (
        <p className="eval-muted">No criteria selected yet.</p>
      )}
      {experiment?.readiness ? (
        <>
          <h3 className="eval-subheading">Verified preparation</h3>
          <div className="eval-table-wrap">
            <table>
              <caption>A/B equality coverage</caption>
              <thead>
                <tr>
                  <th scope="col">Dimension</th>
                  <th scope="col">Policy</th>
                  <th scope="col">A source</th>
                  <th scope="col">B source</th>
                  <th scope="col">Result</th>
                </tr>
              </thead>
              <tbody>
                {experiment.readiness.pins.map((pin) => {
                  const result = pinResult(pin.status);
                  return (
                    <tr key={pin.dimension}>
                      <th scope="row">{pinLabel(pin.dimension)}</th>
                      <td>
                        {pin.requiredEqual ? "Required equal" : "Observed"}
                      </td>
                      <td>
                        {ORIGIN_LABELS[pin.baselineOrigin] ??
                          pin.baselineOrigin}
                      </td>
                      <td>
                        {ORIGIN_LABELS[pin.candidateOrigin] ??
                          pin.candidateOrigin}
                      </td>
                      <td>
                        <StatusChip tone={result.tone} size="sm">
                          {result.label}
                        </StatusChip>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <ul className="eval-criteria" role="list">
            {experiment.readiness.arms.map((arm) => {
              const key = armOf(arm.variantId);
              return (
                <li key={arm.variantId}>
                  {key ? (
                    <ArmKey arm={key}>{key.toUpperCase()}</ArmKey>
                  ) : (
                    <span className="eval-variant-id">{arm.variantId}</span>
                  )}{" "}
                  {arm.eligible}/{arm.expected} eligible; {arm.unsupported}{" "}
                  unsupported; {arm.blocked} blocked. All remain in the
                  denominator.
                </li>
              );
            })}
          </ul>
        </>
      ) : (
        <p className="eval-muted">
          Resolved equality and input compatibility are unverified until
          Prepare. Unknown required-equal pins prevent preparation. Prepare
          creates no Runs or checks; Start is a separate action.
        </p>
      )}
      {experiment?.planSha256 ? (
        <TechnicalDetails description="The frozen plan this experiment runs.">
          <dl className="eval-facts">
            <dt>Prepared plan</dt>
            <dd>
              <IdChip value={experiment.planSha256} label="plan digest" />
            </dd>
            <dt>Revision</dt>
            <dd>{experiment.revision}</dd>
            {experiment.viewSnapshot ? (
              <>
                <dt>View snapshot</dt>
                <dd>
                  <code>{experiment.viewSnapshot}</code>
                </dd>
              </>
            ) : null}
          </dl>
        </TechnicalDetails>
      ) : null}
    </section>
  );
}
