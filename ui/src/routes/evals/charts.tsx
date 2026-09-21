import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { usePublicAPI } from "../../api/context";
import {
  getEvalChart,
  type EvalChart,
  type EvalExperiment,
} from "../../api/evals";
import { ContextLink } from "../../app/context-navigation";
import { EvalError } from "./common";

const CHART_TITLES: Record<EvalChart["chart"], string> = {
  quality: "Quality A/B",
  progress: "Execution progress",
  tokens: "Token distribution",
  duration: "Duration distribution",
  "pair-deltas": "Case differences",
};
const format = (value: number | null | undefined) =>
  value === null || value === undefined
    ? "Unavailable"
    : value.toLocaleString();

function QualityPlot({
  data,
  baseline,
  candidate,
}: {
  data: EvalChart;
  baseline: string;
  candidate: string;
}) {
  const [conditional, setConditional] = useState(false);
  return (
    <>
      <label>
        <input
          type="checkbox"
          checked={conditional}
          onChange={(e) => setConditional(e.target.checked)}
        />
        Quality among scored results only
      </label>
      <p>
        {conditional
          ? "Denominator: scored results"
          : "Denominator: every expected member"}
      </p>
      {[
        [baseline, "A"],
        [candidate, "B"],
      ].map(([id, label]) => {
        const quality = data.quality?.[id!];
        const ratio = conditional
          ? quality?.conditionalQuality
          : quality?.endToEndPass;
        const coverage = data.experimentSummary.counts[id!];
        return (
          <div
            className={`eval-quality-arm eval-arm-${label!.toLowerCase()}`}
            key={id}
          >
            <strong>
              {label}{" "}
              {ratio
                ? `${ratio.numerator}/${ratio.denominator}`
                : "Unavailable"}
            </strong>
            {ratio?.value !== null && ratio?.value !== undefined ? (
              <div className="eval-bar-track" aria-hidden="true">
                <span style={{ width: `${ratio.value * 100}%` }} />
              </div>
            ) : null}
            <small>
              Scored: {coverage?.scored ?? "unavailable"}/
              {coverage?.expected ?? "unavailable"}
            </small>
          </div>
        );
      })}
      <details>
        <summary>Show data table</summary>
        <table>
          <caption>
            {conditional ? "Conditional quality" : "End-to-end quality"}
          </caption>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Passed</th>
              <th>Denominator</th>
              <th>Scored</th>
            </tr>
          </thead>
          <tbody>
            {[
              [baseline, "A"],
              [candidate, "B"],
            ].map(([id, label]) => {
              const q = data.quality?.[id!],
                ratio = conditional ? q?.conditionalQuality : q?.endToEndPass;
              return (
                <tr key={id}>
                  <th>{label}</th>
                  <td>{format(ratio?.numerator)}</td>
                  <td>{format(ratio?.denominator)}</td>
                  <td>{format(data.experimentSummary.counts[id!]?.scored)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </>
  );
}

function DistributionPlot({
  data,
  onBin,
}: {
  data: EvalChart;
  onBin?: (token: string) => void;
}) {
  const bins = data.bins ?? [];
  const maximum = Math.max(
    1,
    ...bins.flatMap((bin) => [bin.counts.a, bin.counts.b]),
  );
  if (!data.coverage.includedPairs)
    return (
      <p>No complete matching pairs are available for this measurement.</p>
    );
  return (
    <>
      <p>
        A: blue, solid · B: orange, striped. Only complete matching pairs.
        p50/p90 describe observed variation.
      </p>
      <div className="eval-histogram">
        {bins.map((bin, index) => (
          <button
            className="eval-bin"
            type="button"
            key={index}
            disabled={!onBin}
            onClick={() => onBin?.(bin.filterToken)}
            aria-label={`${bin.lower} to ${bin.upper} ${data.unit}: A ${bin.counts.a}, B ${bin.counts.b}. Open matching pairs`}
          >
            <span className="eval-bin-bars" aria-hidden="true">
              <i
                className="eval-a"
                style={{ height: `${(bin.counts.a / maximum) * 100}%` }}
              />
              <i
                className="eval-b"
                style={{ height: `${(bin.counts.b / maximum) * 100}%` }}
              />
            </span>
            <small>{format(bin.lower)}</small>
          </button>
        ))}
      </div>
      <table>
        <caption>Observed distributions ({data.unit})</caption>
        <thead>
          <tr>
            <th>Variant</th>
            <th>Samples</th>
            <th>p50</th>
            <th>p90</th>
            <th>Paired total</th>
          </tr>
        </thead>
        <tbody>
          {(["a", "b"] as const).map((arm) => (
            <tr key={arm}>
              <th>{arm.toUpperCase()}</th>
              <td>{format(data.distributions?.[arm].count)}</td>
              <td>{format(data.distributions?.[arm].p50)}</td>
              <td>{format(data.distributions?.[arm].p90)}</td>
              <td>{format(data.distributions?.[arm].total)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details>
        <summary>Show bin data table</summary>
        <table>
          <caption>Shared bins</caption>
          <thead>
            <tr>
              <th>Interval</th>
              <th>A</th>
              <th>B</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {bins.map((bin, index) => (
              <tr key={index}>
                <th>
                  {bin.lower}–{bin.upper}
                  {bin.upperInclusive ? " inclusive" : " exclusive upper"}
                </th>
                <td>{bin.counts.a}</td>
                <td>{bin.counts.b}</td>
                <td>
                  {onBin ? (
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => onBin(bin.filterToken)}
                    >
                      Open matching pairs
                    </button>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

function ProgressPlot({ data }: { data: EvalChart }) {
  const points = data.points ?? [];
  const width = 480,
    height = 220;
  const padding = { left: 38, right: 20, top: 16, bottom: 54 };
  const maxTime = Math.max(1, ...points.map((p) => p.elapsedMs));
  const expected = Math.max(
    1,
    ...Object.values(data.experimentSummary.counts).map((c) => c.expected),
  );
  const x = (time: number) =>
    padding.left + ((width - padding.left - padding.right) * time) / maxTime;
  const y = (count: number) =>
    height -
    padding.bottom -
    ((height - padding.top - padding.bottom) * count) / expected;
  function steps(arm: "a" | "b") {
    let path = "",
      connected = false;
    for (const point of points) {
      const value = point[arm];
      if (value === null) {
        connected = false;
        continue;
      }
      path +=
        !connected || point.gapBefore
          ? ` M ${x(point.elapsedMs)} ${y(value)}`
          : ` H ${x(point.elapsedMs)} V ${y(value)}`;
      connected = true;
    }
    return path;
  }
  return (
    <>
      <p>
        Observed terminal members against experiment wall time. Gaps mean no
        observation, not zero progress.
      </p>
      {points.length ? (
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="A and B observed execution progress; observations follow in the data table"
        >
          <g className="eval-chart-axis" aria-hidden="true">
            {[...new Set([0, Math.floor(expected / 2), expected])].map(
              (count) => (
                <g key={count}>
                  <line x1={x(0)} x2={x(maxTime)} y1={y(count)} y2={y(count)} />
                  <text x={padding.left - 8} y={y(count) + 5} textAnchor="end">
                    {count}
                  </text>
                </g>
              ),
            )}
            <text x={x(0)} y={height - 28}>
              0 s
            </text>
            <text x={x(maxTime)} y={height - 28} textAnchor="end">
              {format(maxTime / 1000)} s
            </text>
            <text x={width / 2} y={height - 4} textAnchor="middle">
              Elapsed time
            </text>
          </g>
          <path d={steps("a")} className="eval-line-a" />
          <path d={steps("b")} className="eval-line-b" />
          {points.map((point, index) => (
            <g key={index}>
              {point.a === null ? null : (
                <circle
                  cx={x(point.elapsedMs)}
                  cy={y(point.a)}
                  r={3}
                  className="eval-dot-a"
                />
              )}
              {point.b === null ? null : (
                <rect
                  x={x(point.elapsedMs) - 3}
                  y={y(point.b) - 3}
                  width={6}
                  height={6}
                  className="eval-dot-b"
                />
              )}
            </g>
          ))}
        </svg>
      ) : (
        <p>No progress observations yet.</p>
      )}
      <p>A: blue circles · B: orange squares and dashed line</p>
      <details>
        <summary>Show data table</summary>
        <table>
          <caption>Observed terminal members</caption>
          <thead>
            <tr>
              <th>Elapsed milliseconds</th>
              <th>A</th>
              <th>B</th>
              <th>Observation gap</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point, index) => (
              <tr key={index}>
                <th>{point.elapsedMs}</th>
                <td>{format(point.a)}</td>
                <td>{format(point.b)}</td>
                <td>
                  {point.gapBefore
                    ? "Gap before this observation"
                    : "No recorded gap"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

function DifferencePlot({ data, id }: { data: EvalChart; id: string }) {
  const points = data.differences ?? [],
    maximum = Math.max(1, ...points.map((p) => Math.abs(p.difference)));
  return (
    <>
      <p>
        Candidate minus baseline ({data.unit}). Negative values use less;
        quality regressions remain labelled separately.
      </p>
      {points.length ? (
        <ul className="eval-differences">
          {points.map((point) => (
            <li key={point.pairId}>
              <ContextLink
                returnLabel="Comparison"
                to={`/evals/experiments/${encodeURIComponent(id)}/pairs/${encodeURIComponent(point.pairId)}?viewSnapshot=${encodeURIComponent(data.viewSnapshot)}`}
              >
                <span>
                  {point.caseId} / {point.sample}
                  {point.regression ? " · Quality regression" : ""}
                </span>
                <span className="eval-difference-track" aria-hidden="true">
                  <i
                    style={{
                      left:
                        point.difference < 0
                          ? `${50 + (point.difference / maximum) * 50}%`
                          : "50%",
                      width: `${(Math.abs(point.difference) / maximum) * 50}%`,
                    }}
                  />
                </span>
                <strong>
                  {point.difference > 0 ? "+" : ""}
                  {format(point.difference)}
                </strong>
              </ContextLink>
            </li>
          ))}
        </ul>
      ) : (
        <p>No complete pairs for this metric.</p>
      )}
      <details>
        <summary>Show data table</summary>
        <table>
          <caption>Case/sample differences</caption>
          <thead>
            <tr>
              <th>Case / sample</th>
              <th>A</th>
              <th>B</th>
              <th>B − A</th>
              <th>Quality</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.pairId}>
                <th>
                  {p.caseId} / {p.sample}
                </th>
                <td>{p.a}</td>
                <td>{p.b}</td>
                <td>{p.difference}</td>
                <td>{p.regression ? "Regression" : "No known regression"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

export function EvalChartPanel({
  experiment,
  chart,
  snapshot,
  metric = "tokens",
  cursor,
  onNext,
  onBin,
  onRefresh,
}: {
  experiment: EvalExperiment;
  chart: EvalChart["chart"];
  snapshot: string;
  metric?: "tokens" | "duration";
  cursor?: string | undefined;
  onNext?: (cursor: string) => void;
  onBin?: (token: string) => void;
  onRefresh?: () => void;
}) {
  const api = usePublicAPI();
  const cache = useQueryClient();
  const query = {
    viewSnapshot: snapshot,
    ...(chart === "pair-deltas"
      ? { metric, limit: 50, ...(cursor ? { cursor } : {}) }
      : {}),
  };
  const result = useQuery({
    queryKey: ["evals", "chart", experiment.experimentId, chart, query],
    queryFn: ({ signal }) =>
      getEvalChart(api, experiment.experimentId, chart, query, signal),
  });
  const data = result.data,
    comparison = experiment.setup?.comparison;
  async function refresh() {
    if (onRefresh) {
      onRefresh();
      return;
    }
    await cache.invalidateQueries({
      queryKey: ["evals", "experiment", experiment.experimentId],
    });
    await cache.invalidateQueries({
      queryKey: ["evals", "chart", experiment.experimentId],
    });
  }
  return (
    <section className="panel eval-panel eval-chart">
      <h3>{CHART_TITLES[chart]}</h3>
      <EvalError error={result.error} reload={() => void refresh()} />
      {result.isPending ? <p role="status">Loading chart…</p> : null}
      {data ? (
        <>
          <p>
            {data.coverage.includedPairs}/{data.coverage.expectedPairs} pairs
            included · {data.freshness}
            {data.measurementScope ? ` · ${data.measurementScope}` : ""}
          </p>
          {chart === "duration" ? (
            <p>
              Execution duration in milliseconds, including queue/hold time;
              Audit duration is the parent interval.
            </p>
          ) : null}
          {Object.entries(data.coverage.reasons).map(([reason, count]) => (
            <p key={reason}>
              {reason.replaceAll("_", " ")}: {count}
            </p>
          ))}
          {chart === "quality" ? (
            <QualityPlot
              data={data}
              baseline={comparison?.baseline ?? "a"}
              candidate={comparison?.candidate ?? "b"}
            />
          ) : chart === "progress" ? (
            <ProgressPlot data={data} />
          ) : chart === "pair-deltas" ? (
            <DifferencePlot data={data} id={experiment.experimentId} />
          ) : (
            <DistributionPlot data={data} {...(onBin ? { onBin } : {})} />
          )}
          {data.page?.hasMore && data.page.nextCursor && onNext ? (
            <button
              type="button"
              className="secondary-button"
              onClick={() => onNext(data.page!.nextCursor!)}
            >
              Next differences
            </button>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
