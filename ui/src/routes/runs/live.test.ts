import { describe, expect, it } from "vitest";

import type {
  PlannerEventData,
  RunEventEnvelope,
} from "../../events/run-events";
import { advancePlannerProjection, type PlannerProjection } from "./live";

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
