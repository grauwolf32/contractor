import type { components } from "../../../api/generated/public";
import type { PerformanceHistory } from "../../../api/performance";
import { formatBytes, formatTimestamp } from "../../artifacts/common";
import { OperationsState } from "../common";
import { MetricSeriesChart, type ChartDatum } from "./chart";
import { performanceFreshnessState } from "./freshness";
import { historyGPUDevices, type GPUColors } from "./gpu-colors";

const measurements = [
  ["utilizationPercent", "GPU utilization", "%"],
  ["memoryUsedBytes", "VRAM used", "MiB"],
  ["memoryTotalBytes", "VRAM capacity", "MiB"],
  ["temperatureCelsius", "Temperature", "°C"],
  ["powerWatts", "Power draw", "W"],
  ["powerLimitWatts", "Power limit", "W"],
] as const;

export function GPUCurrentMetrics({
  gpu,
  readAt,
  colors,
}: {
  gpu: components["schemas"]["PerformanceGPU"] | undefined;
  readAt: string;
  colors: GPUColors;
}) {
  if (gpu === undefined || gpu.freshness.status === "unavailable") return null;
  return [...gpu.devices]
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((device) => {
      const available = measurements.filter(
        ([field]) => device[field] !== undefined,
      );
      if (available.length === 0) return null;
      return (
        <article
          className="performance-metric-card"
          key={device.id}
          aria-label={`GPU ${device.name} · ${device.id.replace(/^GPU-/, "").slice(-8)}`}
        >
          <header>
            <h4 className="performance-device-heading">
              <span
                className="performance-series-swatch"
                style={{ backgroundColor: colors.get(device.id) }}
                aria-hidden="true"
              />
              <span>
                {device.name}
                <small title={device.id}>
                  {device.id.replace(/^GPU-/, "").slice(-8)}
                </small>
              </span>
            </h4>
            <OperationsState
              state={performanceFreshnessState(gpu.freshness, readAt)}
            />
          </header>
          <p className="muted-copy">
            Server host GPU · includes other applications
          </p>
          <dl className="metrics-grid performance-metrics-grid">
            {available.map(([field, label, unit]) => (
              <div key={field}>
                <dt>{label}</dt>
                <dd>
                  {field.startsWith("memory")
                    ? formatBytes(device[field]!)
                    : `${device[field]!.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${unit}`}
                </dd>
              </div>
            ))}
          </dl>
          <small>
            {gpu.freshness.observedAt === undefined
              ? ""
              : `Observed ${formatTimestamp(gpu.freshness.observedAt)} · `}
            <span title={device.id}>
              GPU {device.id.replace(/^GPU-/, "").slice(-8)}
            </span>
          </small>
        </article>
      );
    });
}

export function GPUHistoryCharts({
  history,
  stepSeconds,
  colors,
}: {
  history: PerformanceHistory | undefined;
  stepSeconds: number;
  colors: GPUColors;
}) {
  const points = history?.points ?? [];
  const devices = historyGPUDevices(history);
  return measurements
    .filter(
      ([field]) => field !== "memoryTotalBytes" && field !== "powerLimitWatts",
    )
    .map(([field, label, unit]) => {
      const series = [...devices]
        .map(([id, name]) => {
          const data: ChartDatum[] = points.map((point) => {
            const device = point.gpu?.devices.find(
              (candidate) => candidate.id === id,
            );
            const metric = device?.[field];
            const value = typeof metric === "number" ? metric : metric?.last;
            const observedAt =
              point.kind === "sample" ? point.observedAt : point.minuteStart;
            return {
              observedAt,
              generation: point.generation,
              ...(value === undefined
                ? {}
                : {
                    value: field.startsWith("memory")
                      ? value / (1024 * 1024)
                      : value,
                  }),
            };
          });
          return {
            id,
            label: `${name} · ${id.replace(/^GPU-/, "").slice(-8)}`,
            color: colors.get(id) ?? "var(--accent)",
            data,
          };
        })
        .filter((item) => item.data.some((point) => point.value !== undefined));
      if (series.length === 0) return null;
      return (
        <MetricSeriesChart
          key={field}
          title={label}
          description={`Server host GPUs; ${stepSeconds > 15 ? "last observation in each interval" : "device observation"}, including other applications`}
          series={series}
          expectedStepSeconds={stepSeconds}
          unit={unit}
        />
      );
    });
}
