import { useState } from "react";
import type { EvalExperiment } from "../../api/evals";
import { StatusChip } from "../../ui";
import { EvalChartPanel } from "./charts";
import { ArmKey, EvalField } from "./common";
import { evalConclusionLabel, evalFreshnessText } from "./labels";

const COUNTS = [
  ["terminal", "Terminal"],
  ["scored", "Scored"],
  ["endToEndPassed", "End-to-end passed"],
] as const;

/**
 * The comparison conclusion and each arm's execution and assessment coverage
 * against every expected member. A missing summary is not zero: nothing is
 * concluded until observations arrive.
 */
export function EvalOverviewSummary({
  experiment,
}: {
  experiment: EvalExperiment;
}) {
  const summary = experiment.summary;
  if (!summary)
    return (
      <p className="eval-muted">
        Comparison is waiting for collected observations.
      </p>
    );
  const comparison =
    experiment.setup?.comparison ?? experiment.draft?.comparison;
  const variants = comparison
    ? [comparison.baseline, comparison.candidate]
    : Object.keys(summary.counts);
  const conclusion = evalConclusionLabel(summary.conclusion);
  const freshness = evalFreshnessText(experiment.freshness);
  return (
    <section className="eval-overview-summary" aria-label="Experiment coverage">
      <dl className="eval-conclusion">
        <dt>Conclusion</dt>
        <dd>
          <StatusChip tone={conclusion.tone}>{conclusion.label}</StatusChip>
          {freshness ? (
            <span
              className="eval-conclusion-note"
              data-stale={experiment.freshness === "stale" || undefined}
            >
              {freshness}
            </span>
          ) : null}
        </dd>
      </dl>
      <div className="eval-arms">
        {variants.map((id) => {
          const counts = summary.counts[id];
          if (!counts) return null;
          const arm = comparison
            ? id === comparison.baseline
              ? "a"
              : "b"
            : undefined;
          return (
            <section className="eval-panel eval-arm" data-arm={arm} key={id}>
              <h3>
                {arm ? (
                  <ArmKey arm={arm}>
                    {arm === "a" ? "A · Baseline" : "B · Candidate"}
                  </ArmKey>
                ) : (
                  id
                )}
              </h3>
              <dl className="eval-coverage">
                {COUNTS.map(([key, label]) => (
                  <div key={key}>
                    <dt>{label}</dt>
                    <dd>
                      {counts[key]}
                      <span> / {counts.expected}</span>
                    </dd>
                  </div>
                ))}
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
