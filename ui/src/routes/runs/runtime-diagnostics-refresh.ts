import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { queryKeys } from "../../api/query-keys";
import type { StageAttempt } from "../../api/runs";

function diagnosticQueries(runId: string, stages: ReadonlySet<string>) {
  const runtime = queryKeys.runs.runtimeDiagnosticQueries(runId);
  const tools = queryKeys.runs.toolCallQueries(runId);
  return (query: { queryKey: readonly unknown[] }) =>
    query.queryKey[1] === runId &&
    (query.queryKey[0] === runtime[0] || query.queryKey[0] === tools[0]) &&
    typeof query.queryKey[2] === "string" &&
    stages.has(query.queryKey[2]);
}

/** One RunBody lifetime: entry intent and its own attempt state changes only. */
export function useRuntimeDiagnosticRefresh(
  runId: string,
  attempts: readonly Pick<StageAttempt, "stageExecutionId" | "state">[],
) {
  const client = useQueryClient();
  const previous = useRef(new Map<string, string>());
  const entered = useRef(false);
  // Capture facts that predate this page entry before child widgets mount.
  // Checking data only in the effect could invalidate a just-completed cold
  // request and duplicate the initial read. Later invalidation still filters
  // the two exact diagnostic families, excluding local participant selection.
  const [cachedOnEntry] = useState(() => {
    const stages = new Set(attempts.map((attempt) => attempt.stageExecutionId));
    const selected = diagnosticQueries(runId, stages);
    return new Set(
      client
        .getQueryCache()
        .findAll({
          predicate: (query) =>
            query.state.data !== undefined && selected(query),
        })
        .map((query) => query.queryHash),
    );
  });
  useEffect(() => {
    const next = new Map<string, string>();
    const changed = new Set<string>();
    for (const attempt of attempts) {
      const identity = JSON.stringify([runId, attempt.stageExecutionId]);
      next.set(identity, attempt.state);
      const state = previous.current.get(identity);
      if (state !== undefined && state !== attempt.state) {
        changed.add(attempt.stageExecutionId);
      }
    }
    // Batch by cached identities, rather than scanning QueryCache twice for
    // each attempt in a potentially long complete retained history.
    if (!entered.current && cachedOnEntry.size > 0) {
      void client.invalidateQueries(
        {
          predicate: (query) => cachedOnEntry.has(query.queryHash),
          refetchType: "active",
        },
        { cancelRefetch: false },
      );
    }
    entered.current = true;
    if (changed.size > 0) {
      void client.invalidateQueries({
        predicate: diagnosticQueries(runId, changed),
        refetchType: "active",
      });
    }
    previous.current = next;
  }, [attempts, cachedOnEntry, client, runId]);
}
