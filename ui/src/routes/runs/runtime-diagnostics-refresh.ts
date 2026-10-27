import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { queryKeys } from "../../api/query-keys";
import type { StageAttempt } from "../../api/runs";

/** Lives above live projection remounts; refresh only the changed attempt. */
export function useRuntimeDiagnosticRefresh(
  runId: string,
  attempts: readonly Pick<StageAttempt, "stageExecutionId" | "state">[],
) {
  const client = useQueryClient();
  const previous = useRef(new Map<string, string>());
  useEffect(() => {
    const next = new Map<string, string>();
    for (const attempt of attempts) {
      const identity = JSON.stringify([runId, attempt.stageExecutionId]);
      next.set(identity, attempt.state);
      const state = previous.current.get(identity);
      if (state !== undefined && state !== attempt.state) {
        void client.invalidateQueries({
          queryKey: queryKeys.runs.runtimeDiagnostics(
            runId,
            attempt.stageExecutionId,
          ),
          exact: true,
        });
        void client.invalidateQueries({
          queryKey: queryKeys.runs.toolCalls(runId, attempt.stageExecutionId),
        });
      }
    }
    previous.current = next;
  }, [attempts, client, runId]);
}
