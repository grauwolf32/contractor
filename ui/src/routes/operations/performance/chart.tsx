import { useId } from "react";

import { metricSeriesSegments } from "./series";

export interface ChartDatum {
  observedAt: string;
  generation: string;
  value?: number;
}

function numeric(value: number, unit: string): string {
  const formatted = value.toLocaleString(undefined, {
    maximumFractionDigits: value < 10 ? 2 : 1,
  });
  return unit === "" ? formatted : `${formatted} ${unit}`;
}

export function MetricChart({
  title,
  description,
  data,
  expectedStepSeconds,
  unit,
}: {
  title: string;
  description: string;
  data: readonly ChartDatum[];
  expectedStepSeconds: number;
  unit: string;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const segments = metricSeriesSegments(data, expectedStepSeconds);
  const values = segments.flatMap((segment) => segment.values);
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
  const latest = values.reduce((left, right) =>
    right.time > left.time ? right : left,
  );
  return (
    <article className="performance-chart-card">
      <div className="performance-chart-heading">
        <h4>{title}</h4>
        <strong>{numeric(latest.value, unit)}</strong>
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
          generations are rendered as separate line segments.
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
        {segments.map((segment, index) => (
          <polyline
            className="performance-chart-line"
            key={`${segment.generation}:${segment.values[0]?.time}:${index}`}
            points={segment.values
              .map((value) => `${x(value.time)},${y(value.value)}`)
              .join(" ")}
          />
        ))}
        {values.map((value, index) => (
          <circle
            className="performance-chart-point"
            key={`${value.time}:${value.value}:${index}`}
            cx={x(value.time)}
            cy={y(value.value)}
            r="2.5"
          >
            <title>
              {new Date(value.observedAt).toLocaleString()} ·{" "}
              {numeric(value.value, unit)}
            </title>
          </circle>
        ))}
      </svg>
      <dl
        className="performance-chart-summary"
        aria-label={`${title} numeric summary`}
      >
        <div>
          <dt>Minimum</dt>
          <dd>{numeric(minimum, unit)}</dd>
        </div>
        <div>
          <dt>Latest</dt>
          <dd>{numeric(latest.value, unit)}</dd>
        </div>
        <div>
          <dt>Maximum</dt>
          <dd>{numeric(maximum, unit)}</dd>
        </div>
        <div>
          <dt>Observed points</dt>
          <dd>{values.length.toLocaleString()}</dd>
        </div>
      </dl>
    </article>
  );
}
