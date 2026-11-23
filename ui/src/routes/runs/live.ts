import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { queryKeys } from "../../api/query-keys";
import type { RunStatus } from "../../api/runs";
import { useRunEvents } from "../../events/context";
import type {
  RunEventConnectionState,
  RunResyncReason,
} from "../../events/run-events";
import {
  applyPlannerEvent,
  baselineProjections,
  type PlannerProjection,
} from "./planner-projection";

export interface LiveRunProjection {
  planners: Readonly<Record<string, PlannerProjection>>;
  connection: RunEventConnectionState | "unavailable";
  error?: string;
  resyncReason?: RunResyncReason;
}

export function useLiveRunProjection(run: RunStatus): LiveRunProjection {
  const manager = useRunEvents();
  const queryClient = useQueryClient();
  const [planners, setPlanners] = useState<Record<string, PlannerProjection>>(
    () => baselineProjections(run),
  );
  const plannersRef = useRef(planners);
  const [connection, setConnection] = useState<
    RunEventConnectionState | "unavailable"
  >(run.eventCursor === undefined ? "unavailable" : "connecting");
  const [error, setError] = useState<string | undefined>();
  const [resyncReason, setResyncReason] = useState<
    RunResyncReason | undefined
  >();
  const resyncing = useRef(false);
  const cursorGeneration = run.eventCursor?.generation;
  const cursorSequence = run.eventCursor?.sequence;

  useEffect(() => {
    const eventCursor = run.eventCursor;
    if (eventCursor === undefined) {
      return;
    }
    let active = true;
    const subscription = manager.subscribeRun(run.runId, eventCursor, {
      onPlannerEvent: (event) => {
        const newAttempt = !Object.hasOwn(
          plannersRef.current,
          event.data.stageExecutionId,
        );
        const next = applyPlannerEvent(plannersRef.current, event);
        if (next === undefined) {
          return false;
        }
        plannersRef.current = next;
        if (active) {
          setPlanners(next);
          if (newAttempt) {
            // The attempt's lifecycle event normally started this refetch
            // already; join it rather than restarting it.
            void queryClient.invalidateQueries(
              { queryKey: queryKeys.runs.detail(run.runId), exact: true },
              { cancelRefetch: false },
            );
          }
        }
        return true;
      },
      onLifecycleEvent: () => {
        void queryClient.invalidateQueries({ queryKey: queryKeys.runs.all });
      },
      onResync: (reason) => {
        if (!active || resyncing.current) {
          return;
        }
        resyncing.current = true;
        setResyncReason(reason);
        void queryClient
          .refetchQueries(
            { queryKey: queryKeys.runs.detail(run.runId), exact: true },
            { throwOnError: true },
          )
          .then(() => {
            if (active) {
              resyncing.current = false;
            }
          })
          .catch(() => {
            if (!active) {
              return;
            }
            resyncing.current = false;
            setConnection("error");
            setError("Resynchronization failed; use Refresh to retry.");
          });
      },
      onStateChange: (state) => {
        if (active) {
          setConnection(state);
        }
      },
      onError: (message) => {
        if (active) {
          setError(message);
        }
      },
    });
    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, [
    manager,
    queryClient,
    run.runId,
    run.eventCursor,
    cursorGeneration,
    cursorSequence,
  ]);

  return {
    planners,
    connection,
    ...(error === undefined ? {} : { error }),
    ...(resyncReason === undefined ? {} : { resyncReason }),
  };
}
