import { describe, expect, it } from "vitest";

import type { RunStatus, StageAttempt } from "../../api/runs";
import { deriveRunTriage, formatRunDuration } from "./triage";

const digest = `sha256:${"1".repeat(64)}`;

function attempt(
  stageExecutionId: string,
  stage: string,
  ordinal: number,
): StageAttempt {
  return {
    stageExecutionId,
    stage,
    attempt: ordinal,
    executionConfig: { variant: "base", agents: {} },
    state: "running",
  };
}

function runFixture(): RunStatus {
  return {
    runId: "run-triage",
    workflow: "review@1",
    state: "running",
    deletable: false,
    runtimeLabels: [],
    labels: {},
    runtimeConfiguration: {
      default: {
        label: "default",
        bindingRevision: "1",
        config: { name: "base", version: "1", digest },
      },
      labels: [],
    },
    attempts: [],
    transitions: [],
    outputs: {},
    outputPublications: [],
    createdAt: "2026-09-02T10:00:00Z",
    startedAt: "2026-09-02T10:00:02Z",
    updatedAt: "2026-09-02T10:00:36Z",
  };
}

describe("Run triage", () => {
  it("selects the latest authoritative failure and aggregates bounded metrics", () => {
    const first = attempt("stage-1", "analysis", 1);
    first.state = "interrupted";
    first.termination = {
      outcome: "interrupted",
      code: "worker_lease_lost",
      message: "Worker lease expired.",
      retryable: true,
      phase: "running",
      occurredAt: "2026-09-02T10:00:12Z",
    };
    first.metrics = {
      reportsComplete: true,
      modelCalls: 1,
      inputTokens: 100,
      outputTokens: 50,
      totalTokens: 150,
      toolCalls: 2,
      toolFailures: 0,
      errorCount: 1,
      truncated: false,
    };
    const second = attempt("stage-2", "analysis", 2);
    second.state = "failed";
    second.result = {
      apiVersion: "contractor/v1alpha1",
      outcome: "failed",
      summary: "Planner could not start.",
      artifacts: {},
      error: {
        code: "planner_gateway_unavailable",
        message: "The configured Planner gateway is unavailable.",
        retryable: true,
      },
    };
    second.metrics = {
      reportsComplete: false,
      modelCalls: 2,
      inputTokens: 200,
      outputTokens: 75,
      totalTokens: 275,
      toolCalls: 3,
      toolFailures: 1,
      errorCount: 2,
      truncated: false,
    };
    const run = runFixture();
    run.state = "failed";
    run.attempts = [first, second];
    run.finishedAt = "2026-09-02T10:00:36Z";

    expect(deriveRunTriage(run)).toEqual({
      stage: "analysis",
      stageExecutionId: "stage-2",
      attemptCount: 2,
      durationMs: 34_000,
      outputCount: 0,
      issue: {
        code: "planner_gateway_unavailable",
        message: "The configured Planner gateway is unavailable.",
        retryable: true,
        source: "result",
      },
      metrics: {
        modelCalls: 3,
        totalTokens: 425,
        toolCalls: 5,
        errorCount: 3,
        incomplete: true,
      },
    });
  });

  it("does not present a recovered diagnostic as a successful Run failure", () => {
    const recovered = attempt("stage-1", "analysis", 1);
    recovered.state = "interrupted";
    recovered.diagnostics = {
      items: [
        {
          participant: "worker",
          logicalAgent: "reviewer",
          code: "worker_timeout",
          message: "The first attempt timed out.",
          retryable: true,
        },
      ],
      truncated: false,
    };
    const completed = attempt("stage-2", "publish", 1);
    completed.state = "succeeded";
    const run = runFixture();
    run.state = "succeeded";
    run.attempts = [recovered, completed];
    run.outputs = {
      report: { namespace: "outputs", name: "report", revision: "r1" },
    };

    expect(deriveRunTriage(run)).toMatchObject({
      stage: "publish",
      stageExecutionId: "stage-2",
      attemptCount: 2,
      outputCount: 1,
    });
    expect(deriveRunTriage(run).issue).toBeUndefined();
  });

  it("formats bounded durations for quick scanning", () => {
    expect(formatRunDuration(undefined)).toBe("—");
    expect(formatRunDuration(700)).toBe("<1s");
    expect(formatRunDuration(34_000)).toBe("34s");
    expect(formatRunDuration(254_000)).toBe("4m 14s");
    expect(formatRunDuration(7_440_000)).toBe("2h 4m");
  });
});
