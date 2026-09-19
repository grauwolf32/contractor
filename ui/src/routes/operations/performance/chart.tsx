import { useId } from "react";

import { metricSeriesSegments } from "./series";

export interface ChartDatum {
  observedAt: string;
  generation: string;
  value?: number;
}

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  data: readonly ChartDatum[];
}

interface ChartProps {
  title: string;
  description: string;
  expectedStepSeconds: number;
  unit: string;
}

function numeric(value: number, unit: string): string {
  const formatted = value.toLocaleString(undefined, {
    maximumFractionDigits: value < 10 ? 2 : 1,
  });
  return unit === "" ? formatted : `${formatted} ${unit}`;
}

export function MetricChart({
  data,
  ...props
}: ChartProps & { data: readonly ChartDatum[] }) {
  return (
    <MetricSeriesChart
      {...props}
      showLegend={false}
      series={[
        { id: "metric", label: props.title, color: "var(--accent)", data },
      ]}
    />
  );
}

export function MetricSeriesChart({
  title,
  description,
  series,
  expectedStepSeconds,
  unit,
  showLegend = true,
}: ChartProps & { series: readonly ChartSeries[]; showLegend?: boolean }) {
  const titleId = useId();
  const descriptionId = useId();
  const prepared = series
    .map((item) => {
      const segments = metricSeriesSegments(item.data, expectedStepSeconds);
      const values = segments.flatMap((segment) => segment.values);
      const latest = values.reduce<(typeof values)[number] | undefined>(
        (left, right) =>
          left === undefined || right.time > left.time ? right : left,
        undefined,
      );
      return { ...item, segments, values, latest };
    })
    .filter((item) => item.latest !== undefined);
  const values = prepared.flatMap((item) => item.values);
  if (values.length === 0) {
    return (
      <article className="performance-chart-card">
        <h4>{title}</h4>
        <p className="compact-empty">No observations in this range.</p>
        <small>{description}</small>
      </article>
    );
  }
  const width = 640;
  const height = 190;
  const padding = { left: 48, right: 14, top: 16, bottom: 28 };
  const times = values.map((value) => value.time);
  const numbers = values.map((value) => value.value);
  const firstTime = Math.min(...times);
  const lastTime = Math.max(...times);
  const minimum = Math.min(...numbers);
  const maximum = Math.max(...numbers);
  const yMinimum = minimum === maximum ? Math.max(0, minimum - 1) : minimum;
  const yMaximum = minimum === maximum ? maximum + 1 : maximum;
  const x = (time: number) =>
    padding.left +
    ((time - firstTime) / Math.max(1, lastTime - firstTime)) *
      (width - padding.left - padding.right);
  const y = (value: number) =>
    padding.top +
    (1 - (value - yMinimum) / Math.max(Number.EPSILON, yMaximum - yMinimum)) *
      (height - padding.top - padding.bottom);
  const summaries = prepared.map((item) => (
    <div key={item.id}>
      {showLegend && <h5>{item.label}</h5>}
      <dl
        className="performance-chart-summary"
        aria-label={`${title}${showLegend ? ` · ${item.label}` : ""} numeric summary`}
      >
        <div>
          <dt>Minimum</dt>
          <dd>{numeric(Math.min(...item.values.map((v) => v.value)), unit)}</dd>
        </div>
        <div>
          <dt>Latest</dt>
          <dd>{numeric(item.latest!.value, unit)}</dd>
        </div>
        <div>
          <dt>Maximum</dt>
          <dd>{numeric(Math.max(...item.values.map((v) => v.value)), unit)}</dd>
        </div>
        <div>
          <dt>Observed points</dt>
          <dd>{item.values.length.toLocaleString()}</dd>
        </div>
      </dl>
    </div>
  ));
  return (
    <article className="performance-chart-card">
      <div className="performance-chart-heading">
        <h4>{title}</h4>
        {prepared.length === 1 ? (
          <strong>{numeric(prepared[0]!.latest!.value, unit)}</strong>
        ) : (
          <span className="muted-copy">{prepared.length} GPUs</span>
        )}
      </div>
      <svg
        className="performance-chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        <title id={titleId}>{title}</title>
        <desc id={descriptionId}>
          {description}. Missing observations, long sampling gaps and Server
          generations are rendered as separate line segments.{" "}
          {showLegend &&
            prepared
              .map(
                (item) => `${item.label}: ${numeric(item.latest!.value, unit)}`,
              )
              .join("; ")}
        </desc>
        <line
          className="performance-chart-axis"
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={height - padding.bottom}
        />
        <line
          className="performance-chart-axis"
          x1={padding.left}
          y1={height - padding.bottom}
          x2={width - padding.right}
          y2={height - padding.bottom}
        />
        <text x="4" y={padding.top + 5}>
          {numeric(maximum, unit)}
        </text>
        <text x="4" y={height - padding.bottom}>
          {numeric(minimum, unit)}
        </text>
        {prepared.map((item) => (
          <g key={item.id} data-series-id={item.id}>
            {item.segments.map((segment, index) => (
              <polyline
                className="performance-chart-line"
                style={{ stroke: item.color }}
                key={`${segment.generation}:${segment.values[0]?.time}:${index}`}
                points={segment.values
                  .map((value) => `${x(value.time)},${y(value.value)}`)
                  .join(" ")}
              />
            ))}
            {item.values.map((value, index) => (
              <circle
                className="performance-chart-point"
                style={{ fill: item.color }}
                key={`${value.time}:${value.value}:${index}`}
                cx={x(value.time)}
                cy={y(value.value)}
                r="2.5"
              >
                <title>
                  {showLegend ? `${item.label} · ` : ""}
                  {new Date(value.observedAt).toLocaleString()} ·{" "}
                  {numeric(value.value, unit)}
                </title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      {showLegend && (
        <ul
          className="performance-chart-legend"
          aria-label={`${title} GPU legend`}
        >
          {prepared.map((item) => (
            <li key={item.id} data-series-id={item.id}>
              <span
                className="performance-series-swatch"
                style={{ backgroundColor: item.color }}
                aria-hidden="true"
              />
              <span title={item.id}>{item.label}</span>
              <strong
                title={`Last observed ${new Date(item.latest!.observedAt).toLocaleString()}`}
              >
                {numeric(item.latest!.value, unit)}
              </strong>
            </li>
          ))}
        </ul>
      )}
      {showLegend ? (
        <details className="performance-series-details">
          <summary>Numeric summaries</summary>
          {summaries}
        </details>
      ) : (
        summaries
      )}
    </article>
  );
}
