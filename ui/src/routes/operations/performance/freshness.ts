import type { PerformanceFreshness } from "../../../api/performance";

export function performanceFreshnessState(
  freshness: PerformanceFreshness | undefined,
  readAt: string,
): string {
  if (freshness === undefined) return "unavailable";
  const age = Date.parse(readAt) - Date.parse(freshness.lastAttemptAt);
  if (Number.isFinite(age) && age > freshness.intervalSeconds * 2_000) {
    return "stale";
  }
  return freshness.status;
}
