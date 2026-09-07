import type { ChartDatum } from "./chart";

export interface ChartSegment {
  generation: string;
  values: Array<{ time: number; value: number; observedAt: string }>;
}

export function metricSeriesSegments(
  data: readonly ChartDatum[],
  expectedStepSeconds: number,
): ChartSegment[] {
  const result: ChartSegment[] = [];
  let current: ChartSegment | undefined;
  let previousTime: number | undefined;
  for (const datum of data) {
    const time = Date.parse(datum.observedAt);
    const gap =
      previousTime === undefined
        ? false
        : time - previousTime > expectedStepSeconds * 2_000;
    if (
      datum.value === undefined ||
      !Number.isFinite(datum.value) ||
      datum.value < 0 ||
      !Number.isFinite(time)
    ) {
      current = undefined;
      previousTime = Number.isFinite(time) ? time : undefined;
      continue;
    }
    if (
      current === undefined ||
      current.generation !== datum.generation ||
      gap
    ) {
      current = { generation: datum.generation, values: [] };
      result.push(current);
    }
    current.values.push({
      time,
      value: datum.value,
      observedAt: datum.observedAt,
    });
    previousTime = time;
  }
  return result.filter((segment) => segment.values.length > 0);
}
