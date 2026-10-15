/*
 * The Planner part of a live Run page: per attempt, the subtask plan and the
 * dispatch in flight, advanced by Planner event facts on top of the Run's
 * REST snapshot. A fact that does not follow from the projection (a gap, an
 * unknown attempt) is refused, and the caller resyncs from REST.
 */
import type { PlannerPlan, RunStatus } from "../../api/runs";
import type {
  PlannerEventData,
  PlannerEventKind,
  RunEventEnvelope,
} from "../../events/run-events";

interface DispatchProjection {
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

/** The projections of a Run's REST snapshot, by stage execution ID. */
export function baselineProjections(
  run: RunStatus,
): Record<string, PlannerProjection> {
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

function advancePlannerProjection(
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

/**
 * Applies one Planner fact to the per-attempt projections, or returns
 * `undefined` when it cannot be applied without an authoritative baseline.
 */
export function applyPlannerEvent(
  planners: Readonly<Record<string, PlannerProjection>>,
  event: RunEventEnvelope<PlannerEventData>,
): Record<string, PlannerProjection> | undefined {
  const stageExecutionId = event.data.stageExecutionId;
  // A new attempt's first fact can arrive before the Run refetch that lists
  // the attempt, so it starts an empty projection instead of counting as a
  // gap. Any later fact for an attempt missing from the baseline still is.
  const current = Object.hasOwn(planners, stageExecutionId)
    ? planners[stageExecutionId]
    : event.data.eventKind === "planner.started"
      ? {}
      : undefined;
  if (current === undefined) {
    return undefined;
  }
  const advanced = advancePlannerProjection(current, event);
  if (advanced === undefined) {
    return undefined;
  }
  return { ...planners, [stageExecutionId]: advanced };
}
