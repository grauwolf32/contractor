import type { Query, QueryKey } from "@tanstack/react-query";

import type { PublicAPI } from "../../api/client";
import { queryKeys } from "../../api/query-keys";
import { getRun, type RunStatus } from "../../api/runs";
import { pollInterval } from "../../app/polling";

/** How often a Run that waits for its model route is read again. */
const WAITING_POLL_MS = 20_000;
/** The earliest next read while automatic recovery is due. */
const RECOVERY_POLL_MIN_MS = 2_000;

const whileWaiting = pollInterval<RunStatus>({
  interval: WAITING_POLL_MS,
  settled: (run) => run.state !== "waiting",
});

/**
 * Model recovery (RUN-R30) changes a waiting Run without a Run event: its
 * next automatic check, or the end of automatic recovery, can pass while
 * nothing else happens. The Run is read when the earlier of the two is due,
 * at least 2 s ahead.
 */
function recoveryDelay(run: RunStatus): number {
  const recovery = run.recovery;
  if (recovery === undefined || recovery.requiresRetry) return WAITING_POLL_MS;
  const due = Math.min(
    Date.parse(recovery.nextRetryAt ?? recovery.automaticUntil),
    Date.parse(recovery.automaticUntil),
  );
  return Number.isFinite(due)
    ? Math.max(RECOVERY_POLL_MIN_MS, due - Date.now())
    : RECOVERY_POLL_MIN_MS;
}

/**
 * The one read of a Run's status that the Run page, the Inbox rows and the
 * Inbox detail share: one query key, so they share the cache, and one
 * polling rule. Lifecycle changes reach the Run page through its event
 * stream; the reads repeat only while the Run waits for its model route
 * (every 20 s, or when its next automatic recovery check is due), and stop
 * once it no longer waits. Failed reads back off and a permanent refusal
 * (404) stops them (pollInterval).
 */
export function runStatusQuery(api: PublicAPI, runId: string) {
  return {
    queryKey: queryKeys.runs.detail(runId),
    queryFn: () => getRun(api, runId),
    refetchInterval: (query: Query<RunStatus, Error, RunStatus, QueryKey>) => {
      const next = whileWaiting(query);
      const run = query.state.data;
      return next === false ||
        query.state.status === "error" ||
        run === undefined
        ? next
        : Math.min(next, recoveryDelay(run));
    },
  };
}
