import { useEffect } from "react";
import { queryKeys } from "../../api/query-keys";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { usePublicAPI } from "../../api/context";
import { retryRunGateway, type RunStatus } from "../../api/runs";
import { ErrorNotice, formatTimestamp } from "../artifacts/common";

const recoveryReasons = {
  model_unavailable: "The model was unloaded or is unavailable.",
  gateway_unavailable: "The model gateway is unavailable.",
  gateway_timeout: "The model request timed out.",
  gateway_rate_limited: "The model gateway is rate limiting requests.",
};

export function RunRecoveryControl({ run }: { run: RunStatus }) {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const retry = useMutation({
    mutationFn: () => retryRunGateway(api, run.runId),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: queryKeys.runs.all }),
  });
  const recovery = run.recovery;
  useEffect(() => {
    if (recovery === undefined || recovery.requiresRetry) return;
    // Recovery deadlines can expire without a Run state transition. Refresh at
    // the next advertised check; a one-second floor avoids polling past dates.
    const next = Math.min(
      Date.parse(recovery.nextRetryAt ?? recovery.automaticUntil),
      Date.parse(recovery.automaticUntil),
    );
    const timer = window.setTimeout(
      () =>
        void queryClient.invalidateQueries({
          queryKey: queryKeys.runs.detail(run.runId),
        }),
      Math.max(1000, next - Date.now()),
    );
    return () => window.clearTimeout(timer);
  }, [recovery, queryClient, run.runId]);
  if (recovery === undefined) return null;
  return (
    <div className="run-next-action" role="status">
      <strong>{recoveryReasons[recovery.code]}</strong>
      <p>
        {recovery.requiresRetry
          ? "Automatic recovery has paused. Restore the model and enable retry to continue."
          : recovery.nextRetryAt === undefined
            ? "Waiting for automatic recovery."
            : `Next recovery check: ${formatTimestamp(recovery.nextRetryAt)}.`}
      </p>
      <p>Completed work is retained. You can cancel this Run while it waits.</p>
      {recovery.requiresRetry ? (
        <button
          className="primary-button"
          disabled={retry.isPending}
          onClick={() => retry.mutate()}
        >
          {retry.isPending ? "Enabling retry…" : "Retry model connection"}
        </button>
      ) : null}
      {retry.error === null ? null : <ErrorNotice error={retry.error} />}
    </div>
  );
}
