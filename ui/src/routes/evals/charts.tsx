import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useId, useState, type ReactNode } from "react";
import { usePublicAPI } from "../../api/context";
import {
  getEvalChart,
  type EvalChart,
  type EvalExperiment,
} from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { ArmKey, EvalError } from "./common";

const CHART_TITLES: Record<EvalChart["chart"], string> = {
  quality: "Quality A/B",
  progress: "Execution progress",
  tokens: "Token distribution",
  duration: "Duration distribution",
  "pair-deltas": "Case differences",
};
// A missing measurement is never zero (UUS:178-179).
const format = (value: number | null | undefined) =>
  value === null || value === undefined
    ? "Unavailable"
    : value.toLocaleString();

function DataTable({
  summary = "Show data table",
  children,
}: {
  summary?: string;
  children: ReactNode;
}) {
  return (
    <details className="eval-data-table">
      <summary>{summary}</summary>
      <div className="eval-table-wrap">{children}</div>
    </details>
  );
}

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
      <label className="eval-checkbox">
        <input
          type="checkbox"
          checked={conditional}
          onChange={(e) => setConditional(e.target.checked)}
        />
        <span>Quality among scored results only</span>
      </label>
      <p className="eval-muted">
        {conditional
          ? "Denominator: scored results"
          : "Denominator: every expected member"}
      </p>
      {(
        [
          [baseline, "a"],
          [candidate, "b"],
        ] as const
      ).map(([id, arm]) => {
        const quality = data.quality?.[id];
        const ratio = conditional
          ? quality?.conditionalQuality
          : quality?.endToEndPass;
        const coverage = data.experimentSummary.counts[id];
        return (
          <div className="eval-quality-arm" data-arm={arm} key={arm}>
            <div className="eval-quality-head">
              <ArmKey arm={arm}>{arm.toUpperCase()}</ArmKey>
              <strong>
                {ratio
                  ? `${ratio.numerator}/${ratio.denominator}`
                  : "Unavailable"}
              </strong>
            </div>
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
      <DataTable>
        <table>
          <caption>
            {conditional ? "Conditional quality" : "End-to-end quality"}
          </caption>
          <thead>
            <tr>
              <th scope="col">Variant</th>
              <th scope="col">Passed</th>
              <th scope="col">Denominator</th>
              <th scope="col">Scored</th>
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
                  <th scope="row">{label}</th>
                  <td>{format(ratio?.numerator)}</td>
                  <td>{format(ratio?.denominator)}</td>
                  <td>{format(data.experimentSummary.counts[id!]?.scored)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </DataTable>
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
      <p className="eval-muted">
        No complete matching pairs are available for this measurement.
      </p>
    );
  return (
    <>
      <p className="eval-legend">
        <ArmKey arm="a">A: blue, solid</ArmKey>
        <ArmKey arm="b">B: orange, striped</ArmKey>
        <span>
          Only complete matching pairs. p50/p90 describe observed variation.
        </span>
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
      <div className="eval-table-wrap">
        <table>
          <caption>Observed distributions ({data.unit})</caption>
          <thead>
            <tr>
              <th scope="col">Variant</th>
              <th scope="col">Samples</th>
              <th scope="col">p50</th>
              <th scope="col">p90</th>
              <th scope="col">Paired total</th>
            </tr>
          </thead>
          <tbody>
            {(["a", "b"] as const).map((arm) => (
              <tr key={arm}>
                <th scope="row">{arm.toUpperCase()}</th>
                <td>{format(data.distributions?.[arm].count)}</td>
                <td>{format(data.distributions?.[arm].p50)}</td>
                <td>{format(data.distributions?.[arm].p90)}</td>
                <td>{format(data.distributions?.[arm].total)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <DataTable summary="Show bin data table">
        <table>
          <caption>Shared bins</caption>
          <thead>
            <tr>
              <th scope="col">Interval</th>
              <th scope="col">A</th>
              <th scope="col">B</th>
              <th scope="col">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {bins.map((bin, index) => (
              <tr key={index}>
                <th scope="row">
                  {bin.lower}–{bin.upper}
                  {bin.upperInclusive ? " inclusive" : " exclusive upper"}
                </th>
                <td>{bin.counts.a}</td>
                <td>{bin.counts.b}</td>
                <td>
                  {onBin ? (
                    <button
                      type="button"
                      className="ui-btn"
                      data-size="xs"
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
      </DataTable>
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
      <p className="eval-muted">
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
        <p className="eval-muted">No progress observations yet.</p>
      )}
      <p className="eval-legend">
        <ArmKey arm="a">A: blue circles</ArmKey>
        <ArmKey arm="b">B: orange squares and dashed line</ArmKey>
      </p>
      <DataTable>
        <table>
          <caption>Observed terminal members</caption>
          <thead>
            <tr>
              <th scope="col">Elapsed milliseconds</th>
              <th scope="col">A</th>
              <th scope="col">B</th>
              <th scope="col">Observation gap</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point, index) => (
              <tr key={index}>
                <th scope="row">{point.elapsedMs}</th>
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
      </DataTable>
    </>
  );
}

function DifferencePlot({ data, id }: { data: EvalChart; id: string }) {
  const points = data.differences ?? [],
    maximum = Math.max(1, ...points.map((p) => Math.abs(p.difference)));
  return (
    <>
      <p className="eval-muted">
        Candidate minus baseline ({data.unit}). Negative values use less;
        quality regressions remain labelled separately.
      </p>
      {points.length ? (
        <ul className="eval-differences" role="list">
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
        <p className="eval-muted">No complete pairs for this metric.</p>
      )}
      <DataTable>
        <table>
          <caption>Case/sample differences</caption>
          <thead>
            <tr>
              <th scope="col">Case / sample</th>
              <th scope="col">A</th>
              <th scope="col">B</th>
              <th scope="col">B − A</th>
              <th scope="col">Quality</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.pairId}>
                <th scope="row">
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
      </DataTable>
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
  const heading = useId();
  const query = {
    viewSnapshot: snapshot,
    ...(chart === "pair-deltas"
      ? { metric, limit: 50, ...(cursor ? { cursor } : {}) }
      : {}),
  };
  const result = useQuery({
    queryKey: queryKeys.evals.chart(experiment.experimentId, chart, query),
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
      queryKey: queryKeys.evals.experiment(experiment.experimentId),
    });
    await cache.invalidateQueries({
      queryKey: queryKeys.evals.projection("chart", experiment.experimentId),
    });
  }
  return (
    <section className="eval-panel eval-chart" aria-labelledby={heading}>
      <h3 id={heading}>{CHART_TITLES[chart]}</h3>
      <EvalError error={result.error} reload={() => void refresh()} />
      {result.isPending ? <p role="status">Loading chart…</p> : null}
      {data ? (
        <>
          <p className="eval-chart-scope">
            {data.coverage.includedPairs}/{data.coverage.expectedPairs} pairs
            included · {data.freshness}
            {data.measurementScope ? ` · ${data.measurementScope}` : ""}
          </p>
          {chart === "duration" ? (
            <p className="eval-muted">
              Execution duration in milliseconds, including queue/hold time; a
              check&apos;s duration is its parent interval.
            </p>
          ) : null}
          {Object.entries(data.coverage.reasons).map(([reason, count]) => (
            <p className="eval-muted" key={reason}>
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
            <div className="eval-actions">
              <button
                type="button"
                className="ui-btn"
                data-size="sm"
                onClick={() => onNext(data.page!.nextCursor!)}
              >
                Next differences
              </button>
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
