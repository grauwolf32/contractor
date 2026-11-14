/*
 * One polling rule for Server state that changes on its own (AGENTS.md 3.8):
 * poll at a base interval while the object can still change, stop (or slow
 * down) once it has settled, and back off exponentially while reads fail.
 * pollInterval returns a TanStack Query `refetchInterval` function.
 */
import { PublicAPIError } from "../api/error";

/** The part of a TanStack Query that the interval reads. */
interface PolledQuery<T> {
  state: {
    data: T | undefined;
    error: Error | null;
    status: "pending" | "error" | "success";
    dataUpdateCount: number;
    errorUpdateCount: number;
  };
}

interface PollOptions<T> {
  /** Milliseconds between reads while the data can still change. */
  interval: number;
  /** True once the data can no longer change on its own (a terminal state). */
  settled?: (data: T) => boolean;
  /** What happens once settled: false (the default) stops; a number slows down. */
  settledInterval?: number | false;
  /** Longest wait between failed reads; default 60 seconds. */
  maxInterval?: number;
  /** A failed continuation can retain partial data instead of throwing. */
  readError?: (data: T) => Error | null;
}

const DEFAULT_MAX_INTERVAL = 60_000;

/** Count each completed read once, including a partial result with an error. */
const lastRead = new WeakMap<
  object,
  { dataUpdateCount: number; errorUpdateCount: number; failures: number }
>();

/** Failed reads since the query last read data successfully. */
function failedReads(query: PolledQuery<unknown>, error: Error | null): number {
  const { dataUpdateCount, errorUpdateCount } = query.state;
  const previous = lastRead.get(query);
  if (
    previous !== undefined &&
    previous.dataUpdateCount === dataUpdateCount &&
    previous.errorUpdateCount === errorUpdateCount
  )
    return previous.failures;
  const failures = error === null ? 0 : (previous?.failures ?? 0) + 1;
  lastRead.set(query, { dataUpdateCount, errorUpdateCount, failures });
  return failures;
}

/**
 * A refusal that reading again cannot change: a client error other than a
 * timeout or rate limit, such as 404 for a resource that is gone.
 */
function permanent(error: unknown): boolean {
  return (
    error instanceof PublicAPIError &&
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 429
  );
}

/**
 * A `refetchInterval` for a query whose data changes on its own:
 *
 * - while reads fail, the wait doubles per failed read (2x, 4x, … the base
 *   interval) up to `maxInterval`, and returns to the base after a success;
 *   a permanent refusal (a 4xx other than 408 and 429, such as 404) stops
 *   polling;
 * - once `settled(data)` holds, polling stops, or continues at
 *   `settledInterval`;
 * - otherwise, also before the first read, it is the base `interval`.
 *
 * ```ts
 * useQuery({
 *   queryKey: queryKeys.runs.detail(runId),
 *   queryFn: () => getRun(api, runId),
 *   refetchInterval: pollInterval({
 *     interval: 2_000,
 *     settled: (run) => isTerminalRunState(run.state),
 *   }),
 * });
 * ```
 *
 * Reaching the settled state does not refresh anything else: a page that
 * shows related data (outputs, reports) invalidates those queries when it
 * sees the transition.
 */
export function pollInterval<T>(
  options: PollOptions<T>,
): (query: PolledQuery<T>) => number | false {
  const {
    interval,
    settled,
    settledInterval = false,
    maxInterval = DEFAULT_MAX_INTERVAL,
    readError,
  } = options;
  return (query) => {
    const data = query.state.data;
    const error =
      query.state.error ??
      (data === undefined ? null : (readError?.(data) ?? null));
    const failures = failedReads(query, error);
    if (failures > 0)
      return permanent(error)
        ? false
        : Math.min(maxInterval, interval * 2 ** failures);
    if (data !== undefined && settled?.(data) === true) return settledInterval;
    return interval;
  };
}
