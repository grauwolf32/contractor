import { useState } from "react";
import type { EvalExperiment } from "../../api/evals";
import { EvalChartPanel } from "./charts";
import { EvalField } from "./common";

export function EvalOverviewSummary({
  experiment,
}: {
  experiment: EvalExperiment;
}) {
  const summary = experiment.summary;
  if (!summary) return <p>Comparison is waiting for collected observations.</p>;
  const comparison =
    experiment.setup?.comparison ?? experiment.draft?.comparison;
  const variants = comparison
    ? [comparison.baseline, comparison.candidate]
    : Object.keys(summary.counts);
  return (
    <section className="eval-overview-summary" aria-label="Experiment coverage">
      <p className="eval-conclusion" data-conclusion={summary.conclusion}>
        <strong>Conclusion: {summary.conclusion}</strong>
        <span>
          {experiment.freshness === "stale"
            ? "Evidence changed · refresh to review the current result"
            : experiment.freshness}
        </span>
      </p>
      <div className="eval-variants">
        {variants.map((id) => {
          const counts = summary.counts[id];
          if (!counts) return null;
          const arm = comparison
            ? id === comparison.baseline
              ? "a"
              : "b"
            : undefined;
          return (
            <section
              className={`panel eval-panel ${arm ? `eval-arm-${arm}` : ""}`}
              key={id}
            >
              <h3>
                {arm === "a"
                  ? "A · Baseline"
                  : arm === "b"
                    ? "B · Candidate"
                    : id}
              </h3>
              <dl className="eval-coverage">
                <div>
                  <dt>Terminal</dt>
                  <dd>
                    {counts.terminal}
                    <span> / {counts.expected}</span>
                  </dd>
                </div>
                <div>
                  <dt>Scored</dt>
                  <dd>
                    {counts.scored}
                    <span> / {counts.expected}</span>
                  </dd>
                </div>
                <div>
                  <dt>End-to-end passed</dt>
                  <dd>
                    {counts.endToEndPassed}
                    <span> / {counts.expected}</span>
                  </dd>
                </div>
              </dl>
            </section>
          );
        })}
      </div>
    </section>
  );
}

export function EvalOverviewCharts({
  experiment,
  snapshot,
}: {
  experiment: EvalExperiment;
  snapshot: string;
}) {
  const [selected, setSelected] = useState<"quality" | "progress">("quality");
  return (
    <>
      <div className="eval-overview-selector">
        <EvalField label="Overview chart">
          <select
            value={selected}
            onChange={(event) =>
              setSelected(
                event.target.value === "progress" ? "progress" : "quality",
              )
            }
          >
            <option value="quality">Quality A/B</option>
            <option value="progress">Execution progress</option>
          </select>
        </EvalField>
      </div>
      <div className="eval-overview-charts">
        {(["quality", "progress"] as const).map((chart) => (
          <div key={chart} data-selected={chart === selected}>
            <EvalChartPanel
              experiment={experiment}
              chart={chart}
              snapshot={snapshot}
            />
          </div>
        ))}
      </div>
    </>
  );
}
