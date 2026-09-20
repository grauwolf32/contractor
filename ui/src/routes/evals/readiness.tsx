import type { EvalDraft, EvalExperiment } from "../../api/evals";

export function EvalReadiness({
  draft,
  experiment,
}: {
  draft?: EvalDraft | undefined;
  experiment?: EvalExperiment | undefined;
}) {
  const setup = draft ?? experiment?.setup;
  if (!setup) return null;
  const expected = draft
    ? draft.caseIds.length * 2 * draft.repetitions
    : experiment?.expectedMembers;
  return (
    <section className="eval-readiness">
      <h2>Readiness and exact setup</h2>
      <p>
        <strong>{expected} expected members</strong>
        {setup.caseIds && setup.repetitions
          ? ` · ${setup.caseIds.length} cases × 2 variants × ${setup.repetitions} repetitions`
          : ""}
      </p>
      <div className="eval-variants">
        {[setup.comparison.baseline, setup.comparison.candidate]
          .map((id) => setup.variants.find((variant) => variant.id === id)!)
          .map((v) => (
            <div
              className={`panel eval-panel eval-arm-${v.id === setup.comparison.baseline ? "a" : "b"}`}
              key={v.id}
            >
              <h3>
                {v.id === setup.comparison.baseline
                  ? "A · Baseline"
                  : "B · Candidate"}
              </h3>
              <p>
                {v.kind}: {v.selector}
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
                <details>
                  <summary>Exact execution overrides</summary>
                  <pre>{JSON.stringify(v.executionConfig, null, 2)}</pre>
                </details>
              ) : null}
            </div>
          ))}
      </div>
      <dl className="eval-facts">
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
        <dd>{setup.comparison.requiredEqual.join(", ") || "None declared"}</dd>
        <dt>Allowed differences</dt>
        <dd>
          {setup.comparison.allowedDifferences.join(", ") || "None declared"}
        </dd>
      </dl>
      <h3>Assessment checks</h3>
      <ul>
        {setup.checks.map((check) => (
          <li key={check.id}>
            {check.id} · {check.evaluator} ·{" "}
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
      {experiment?.readiness ? (
        <>
          <h3>Verified preparation</h3>
          <div className="eval-table-wrap">
            <table>
              <caption>A/B equality coverage</caption>
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th>Policy</th>
                  <th>A source</th>
                  <th>B source</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {experiment.readiness.pins.map((pin) => (
                  <tr key={pin.dimension}>
                    <th>{pin.dimension}</th>
                    <td>{pin.requiredEqual ? "Required equal" : "Observed"}</td>
                    <td>{pin.baselineOrigin}</td>
                    <td>{pin.candidateOrigin}</td>
                    <td>{pin.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {experiment.readiness.arms.map((arm) => (
            <p key={arm.variantId}>
              {arm.variantId}: {arm.eligible}/{arm.expected} eligible;{" "}
              {arm.unsupported} unsupported; {arm.blocked} blocked. All remain
              in the denominator.
            </p>
          ))}
        </>
      ) : (
        <p>
          Resolved equality and input compatibility are unverified until
          Prepare. Unknown required-equal pins prevent preparation. Prepare
          creates no Runs or Audits; Start is a separate action.
        </p>
      )}
      {experiment?.planSha256 ? (
        <p className="eval-digest">
          Prepared plan: <code>{experiment.planSha256}</code>
        </p>
      ) : null}
    </section>
  );
}
