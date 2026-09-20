import { useState } from "react";
import type { EvalExperiment } from "../../api/evals";
import { EvalChartPanel } from "./charts";
import { EvalField } from "./common";

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
