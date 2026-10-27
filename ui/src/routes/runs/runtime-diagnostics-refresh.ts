import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { queryKeys } from "../../api/query-keys";
import type { StageAttempt } from "../../api/runs";

/** One RunBody lifetime: entry intent and its own attempt state changes only. */
export function useRuntimeDiagnosticRefresh(
  runId: string,
  attempts: readonly Pick<StageAttempt, "stageExecutionId" | "state">[],
) {
  const client = useQueryClient();
  const previous = useRef(new Map<string, string>());
  // Capture facts that predate this page entry before child widgets mount.
  // Checking data only in the effect could invalidate a just-completed cold
  // request and duplicate the initial read. Later invalidation still filters
  // the two exact diagnostic families, excluding local participant selection.
  const [cachedOnEntry] = useState(() => {
    const stages = new Set(attempts.map((attempt) => attempt.stageExecutionId));
    return new Set(
      client
        .getQueryCache()
        .findAll({
          predicate: (query) =>
            query.state.data !== undefined &&
            query.queryKey[1] === runId &&
            typeof query.queryKey[2] === "string" &&
            stages.has(query.queryKey[2]),
        })
        .map((query) => query.queryHash),
    );
  });
  useEffect(() => {
    const next = new Map<string, string>();
    for (const attempt of attempts) {
      const identity = JSON.stringify([runId, attempt.stageExecutionId]);
      next.set(identity, attempt.state);
      const state = previous.current.get(identity);
      if (state === undefined || state !== attempt.state) {
        const entry = state === undefined;
        const cached = (query: { queryHash: string }) =>
          !entry || cachedOnEntry.has(query.queryHash);
        void client.invalidateQueries(
          {
            queryKey: queryKeys.runs.runtimeDiagnostics(
              runId,
              attempt.stageExecutionId,
            ),
            exact: true,
            predicate: cached,
            refetchType: "active",
          },
          { cancelRefetch: !entry },
        );
        void client.invalidateQueries(
          {
            queryKey: queryKeys.runs.toolCalls(runId, attempt.stageExecutionId),
            predicate: cached,
            refetchType: "active",
          },
          { cancelRefetch: !entry },
        );
      }
    }
    previous.current = next;
  }, [attempts, cachedOnEntry, client, runId]);
}
