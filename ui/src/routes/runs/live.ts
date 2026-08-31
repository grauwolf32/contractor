import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { queryKeys } from "../../api/query-keys";
import type { PlannerPlan, RunStatus } from "../../api/runs";
import { useRunEvents } from "../../events/context";
import type {
  PlannerEventData,
  PlannerEventKind,
  RunEventConnectionState,
  RunEventEnvelope,
  RunResyncReason,
} from "../../events/run-events";

export interface DispatchProjection {
  callId: string;
  subtaskId: string;
  workerName: string;
  phase: "selected" | "running";
}

export interface PlannerProjection {
  plan?: PlannerPlan;
  dispatch?: DispatchProjection;
  lastEventKind?: PlannerEventKind;
  lastOccurredAt?: string;
}

export interface LiveRunProjection {
  planners: Readonly<Record<string, PlannerProjection>>;
  connection: RunEventConnectionState | "unavailable";
  error?: string;
  resyncReason?: RunResyncReason;
}

function baseline(run: RunStatus): Record<string, PlannerProjection> {
  return Object.fromEntries(
    run.attempts.map((attempt) => [
      attempt.stageExecutionId,
      attempt.plan === undefined ? {} : { plan: attempt.plan },
    ]),
  );
}

function subtaskExists(plan: PlannerPlan, subtaskId: string): boolean {
  return plan.subtasks.some((subtask) => subtask.id === subtaskId);
}

function dispatchFromEvent(
  event: PlannerEventData,
  phase: DispatchProjection["phase"],
): DispatchProjection | undefined {
  if (
    event.callId === undefined ||
    event.subtaskId === undefined ||
    event.workerName === undefined
  ) {
    return undefined;
  }
  return {
    callId: event.callId,
    subtaskId: event.subtaskId,
    workerName: event.workerName,
    phase,
  };
}

function advancesPlanRevision(
  current: PlannerPlan | undefined,
  next: PlannerPlan,
): boolean {
  return next.revision === (current?.revision ?? 0) + 1;
}

export function advancePlannerProjection(
  current: PlannerProjection,
  event: RunEventEnvelope<PlannerEventData>,
): PlannerProjection | undefined {
  const data = event.data;
  const next: PlannerProjection = {
    ...current,
    lastEventKind: data.eventKind,
    lastOccurredAt: event.occurredAt,
  };
  switch (data.eventKind) {
    case "planner.plan_changed":
      if (
        data.plan === undefined ||
        current.plan?.activeDispatch !== undefined ||
        !advancesPlanRevision(current.plan, data.plan)
      ) {
        return undefined;
      }
      next.plan = data.plan;
      delete next.dispatch;
      return next;
    case "planner.current_changed":
      if (
        current.plan === undefined ||
        data.planRevision !== current.plan.revision ||
        (data.subtaskId !== undefined &&
          !subtaskExists(current.plan, data.subtaskId))
      ) {
        return undefined;
      }
      if (data.subtaskId === undefined) {
        const withoutCurrent = { ...current.plan };
        delete withoutCurrent.currentSubtaskId;
        next.plan = withoutCurrent;
      } else {
        next.plan = { ...current.plan, currentSubtaskId: data.subtaskId };
      }
      return next;
    case "planner.dispatch_selected": {
      const dispatch = dispatchFromEvent(data, "selected");
      if (
        current.plan === undefined ||
        data.planRevision !== current.plan.revision ||
        dispatch === undefined ||
        current.plan.currentSubtaskId !== dispatch.subtaskId ||
        current.plan.activeDispatch !== undefined
      ) {
        return undefined;
      }
      next.dispatch = dispatch;
      return next;
    }
    case "planner.dispatch_started": {
      const dispatch = dispatchFromEvent(data, "running");
      if (
        data.plan === undefined ||
        dispatch === undefined ||
        !advancesPlanRevision(current.plan, data.plan)
      ) {
        return undefined;
      }
      next.plan = data.plan;
      next.dispatch = dispatch;
      return next;
    }
    case "planner.dispatch_completed": {
      const activeDispatch = current.plan?.activeDispatch;
      if (
        data.plan === undefined ||
        data.callId === undefined ||
        data.subtaskId === undefined ||
        data.workerName === undefined ||
        (data.outcome !== "succeeded" && data.outcome !== "failed") ||
        activeDispatch === undefined ||
        activeDispatch.callId !== data.callId ||
        activeDispatch.subtaskId !== data.subtaskId ||
        activeDispatch.workerName !== data.workerName ||
        !advancesPlanRevision(current.plan, data.plan)
      ) {
        return undefined;
      }
      next.plan = data.plan;
      delete next.dispatch;
      return next;
    }
    case "planner.finish_requested":
      if (
        data.planRevision === undefined ||
        (data.planRevision !== 0 &&
          (current.plan === undefined ||
            current.plan.revision !== data.planRevision))
      ) {
        return undefined;
      }
      return next;
    case "planner.started":
    case "planner.request_recorded":
    case "planner.activity":
    case "planner.completed":
    case "planner.failed":
      return next;
  }
}

export function useLiveRunProjection(run: RunStatus): LiveRunProjection {
  const manager = useRunEvents();
  const queryClient = useQueryClient();
  const [planners, setPlanners] = useState<Record<string, PlannerProjection>>(
    () => baseline(run),
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
        const stageExecutionId = event.data.stageExecutionId;
        const current = plannersRef.current[stageExecutionId];
        if (current === undefined) {
          return false;
        }
        const advanced = advancePlannerProjection(current, event);
        if (advanced === undefined) {
          return false;
        }
        const next = { ...plannersRef.current, [stageExecutionId]: advanced };
        plannersRef.current = next;
        if (active) {
          setPlanners(next);
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
            setError(
              "Authoritative REST resynchronization failed; use Refresh to retry.",
            );
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
