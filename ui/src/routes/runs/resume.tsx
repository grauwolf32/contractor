import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import { resumeRun, type RunStatus } from "../../api/runs";
import { ErrorNotice } from "../artifacts/common";

export function RunResumeControl({ run }: { run: RunStatus }) {
  const api = usePublicAPI();
  const client = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const inFlight = useRef(false);
  const source = run.resumeStageExecutionId;
  const stage = run.attempts.find(
    (attempt) => attempt.stageExecutionId === source,
  );
  const mutation = useMutation({
    mutationFn: (sourceID: string) => resumeRun(api, run.runId, sourceID),
    onSuccess: () => setConfirming(false),
    onSettled: async () => {
      try {
        await Promise.all([
          client.invalidateQueries({ queryKey: queryKeys.runs.all }),
          client.invalidateQueries({ queryKey: queryKeys.queue.all }),
          ...(run.projectId === undefined
            ? []
            : [
                client.invalidateQueries({
                  queryKey: queryKeys.projects.detail(run.projectId),
                }),
              ]),
        ]);
      } finally {
        inFlight.current = false;
      }
    },
  });
  if (run.state !== "failed") return null;
  return (
    <section className="panel" aria-label="Continue failed Run">
      <h3>Continue from failed stage</h3>
      <p>
        Successful stages and their results are preserved. The failed stage gets
        a new attempt with its saved inputs and configuration.
      </p>
      {source === undefined ? (
        <p className="muted-copy">
          Continuation is unavailable: cleanup must finish and the Run must have
          a failed stage. Audit-managed and evaluation Runs use their own
          lifecycle controls.
        </p>
      ) : confirming ? (
        <div role="group" aria-label="Confirm continuation">
          <p>
            Retry stage <strong>{stage?.stage ?? source}</strong>? Model and
            tool calls will run again and may repeat external side effects.
          </p>
          <button
            type="button"
            disabled={mutation.isPending}
            onClick={() => {
              if (inFlight.current) return;
              inFlight.current = true;
              mutation.mutate(source);
            }}
          >
            {mutation.isPending ? "Continuing…" : "Confirm continuation"}
          </button>
          <button
            type="button"
            disabled={mutation.isPending}
            onClick={() => setConfirming(false)}
          >
            Back
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => {
            mutation.reset();
            setConfirming(true);
          }}
        >
          Continue from failed stage
        </button>
      )}
      {mutation.error === null ? null : <ErrorNotice error={mutation.error} />}
    </section>
  );
}
