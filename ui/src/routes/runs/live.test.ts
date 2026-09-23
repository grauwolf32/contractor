import { describe, expect, it } from "vitest";

import type {
  PlannerEventData,
  RunEventEnvelope,
} from "../../events/run-events";
import {
  advancePlannerProjection,
  applyPlannerEvent,
  type PlannerProjection,
} from "./live";

function event(data: PlannerEventData): RunEventEnvelope<PlannerEventData> {
  return {
    cursor: { generation: "run-generation-1", sequence: "2" },
    occurredAt: "2026-08-31T12:00:00Z",
    data,
  };
}

const identity = {
  stageExecutionId: "stage-1",
  sessionId: "session-1",
  invocationId: "invocation-1",
};

const initial: PlannerProjection = {
  plan: {
    revision: 1,
    subtasks: [
      {
        id: "0",
        objective: "Inspect source",
        instructions: "Use bounded source tools",
        status: "pending",
      },
    ],
    currentSubtaskId: "0",
  },
};

describe("Planner live projection", () => {
  it("accepts only the next durable plan revision", () => {
    const changed = {
      revision: 2,
      subtasks: [
        ...initial.plan!.subtasks,
        {
          id: "1",
          objective: "Review boundaries",
          instructions: "Inspect the trust boundaries",
          status: "pending" as const,
        },
      ],
      currentSubtaskId: "0",
    };
    expect(
      advancePlannerProjection(
        initial,
        event({
          ...identity,
          eventKind: "planner.plan_changed",
          plan: changed,
        }),
      )?.plan,
    ).toEqual(changed);
    expect(
      advancePlannerProjection(
        initial,
        event({
          ...identity,
          eventKind: "planner.plan_changed",
          plan: { ...changed, revision: 3 },
        }),
      ),
    ).toBeUndefined();
  });

  it("rejects a logical Worker selection for a non-current subtask", () => {
    expect(
      advancePlannerProjection(
        {
          plan: {
            revision: 2,
            subtasks: [
              ...initial.plan!.subtasks,
              {
                id: "1",
                objective: "Review boundaries",
                instructions: "Inspect the trust boundaries",
                status: "pending",
              },
            ],
            currentSubtaskId: "0",
          },
        },
        event({
          ...identity,
          eventKind: "planner.dispatch_selected",
          planRevision: 2,
          subtaskId: "1",
          callId: "dispatch-0001",
          workerName: "reviewer",
        }),
      ),
    ).toBeUndefined();
  });

  it("clears the active Worker when its dispatch completes", () => {
    const active: PlannerProjection = {
      plan: {
        revision: 2,
        subtasks: [
          {
            ...initial.plan!.subtasks[0]!,
            status: "running",
          },
        ],
        currentSubtaskId: "0",
        activeDispatch: {
          callId: "dispatch-0001",
          subtaskId: "0",
          workerName: "reviewer",
        },
      },
      dispatch: {
        callId: "dispatch-0001",
        subtaskId: "0",
        workerName: "reviewer",
        phase: "running",
      },
    };
    const completed = advancePlannerProjection(
      active,
      event({
        ...identity,
        eventKind: "planner.dispatch_completed",
        plan: {
          revision: 3,
          subtasks: [
            {
              ...initial.plan!.subtasks[0]!,
              status: "succeeded",
            },
          ],
        },
        subtaskId: "0",
        callId: "dispatch-0001",
        workerName: "reviewer",
        outcome: "succeeded",
      }),
    );
    expect(completed?.plan?.subtasks[0]?.status).toBe("succeeded");
    expect(completed?.dispatch).toBeUndefined();
  });
});

describe("Planner projections per attempt", () => {
  it("starts a new attempt from its first Planner fact", () => {
    const planners = { "stage-1": initial };
    const next = applyPlannerEvent(
      planners,
      event({
        ...identity,
        stageExecutionId: "stage-2",
        eventKind: "planner.started",
      }),
    );
    expect(next).toEqual({
      "stage-1": initial,
      "stage-2": {
        lastEventKind: "planner.started",
        lastOccurredAt: "2026-08-31T12:00:00Z",
      },
    });
    expect(planners).toEqual({ "stage-1": initial });
  });

  it("still requires a baseline for a later fact of an unknown attempt", () => {
    expect(
      applyPlannerEvent(
        { "stage-1": initial },
        event({
          ...identity,
          stageExecutionId: "stage-2",
          eventKind: "planner.activity",
        }),
      ),
    ).toBeUndefined();
    expect(
      applyPlannerEvent(
        {},
        event({
          ...identity,
          stageExecutionId: "constructor",
          eventKind: "planner.activity",
        }),
      ),
    ).toBeUndefined();
  });

  it("advances a known attempt and rejects a fact it cannot apply", () => {
    expect(
      applyPlannerEvent(
        { "stage-1": initial },
        event({ ...identity, eventKind: "planner.activity" }),
      )?.["stage-1"],
    ).toEqual({
      ...initial,
      lastEventKind: "planner.activity",
      lastOccurredAt: "2026-08-31T12:00:00Z",
    });
    expect(
      applyPlannerEvent(
        { "stage-1": initial },
        event({
          ...identity,
          eventKind: "planner.current_changed",
          planRevision: 9,
          subtaskId: "0",
        }),
      ),
    ).toBeUndefined();
  });
});
