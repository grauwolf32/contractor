import type { components } from "../../api/generated/public";
import type { RunStatus, StageAttempt } from "../../api/runs";

type AttemptDiagnostic = components["schemas"]["AttemptDiagnostic"];

export interface RunTriageIssue {
  code: string;
  message: string;
  retryable?: boolean;
  participant?: AttemptDiagnostic["participant"];
  logicalAgent?: string;
  source: "result" | "termination" | "diagnostic" | "cancellation";
}

export interface RunTriageMetrics {
  modelCalls: number;
  totalTokens: number;
  toolCalls: number;
  errorCount: number;
  incomplete: boolean;
}

export interface RunTriage {
  stage?: string;
  stageExecutionId?: string;
  attemptCount: number;
  durationMs?: number;
  outputCount: number;
  issue?: RunTriageIssue;
  metrics?: RunTriageMetrics;
}

function latestAttempt(run: RunStatus): StageAttempt | undefined {
  if (run.activeStageExecutionId !== undefined) {
    const active = run.attempts.find(
      (attempt) => attempt.stageExecutionId === run.activeStageExecutionId,
    );
    if (active !== undefined) {
      return active;
    }
  }
  return run.attempts.at(-1);
}

function issueFromAttempt(attempt: StageAttempt): RunTriageIssue | undefined {
  if (attempt.result?.error !== undefined) {
    return {
      code: attempt.result.error.code,
      message: attempt.result.error.message,
      retryable: attempt.result.error.retryable,
      source: "result",
    };
  }
  if (attempt.termination !== undefined) {
    return {
      code: attempt.termination.code,
      message: attempt.termination.message,
      retryable: attempt.termination.retryable,
      source: "termination",
    };
  }
  const diagnostic = attempt.diagnostics?.items.at(-1);
  if (diagnostic === undefined) {
    return undefined;
  }
  return {
    code: diagnostic.code,
    message: diagnostic.message,
    ...(diagnostic.retryable === undefined
      ? {}
      : { retryable: diagnostic.retryable }),
    participant: diagnostic.participant,
    ...(diagnostic.logicalAgent === undefined
      ? {}
      : { logicalAgent: diagnostic.logicalAgent }),
    source: "diagnostic",
  };
}

function issueContext(run: RunStatus): {
  attempt: StageAttempt | undefined;
  issue: RunTriageIssue | undefined;
} {
  if (run.state === "cancelled" || run.state === "cancelling") {
    if (run.cancellation !== undefined) {
      return {
        attempt: latestAttempt(run),
        issue: {
          code: run.cancellation.code,
          message: run.cancellation.reason ?? "Cancellation was requested.",
          retryable: false,
          source: "cancellation",
        },
      };
    }
  }

  if (run.state !== "failed" && run.state !== "cancelled") {
    return { attempt: latestAttempt(run), issue: undefined };
  }

  for (let index = run.attempts.length - 1; index >= 0; index -= 1) {
    const attempt = run.attempts[index];
    if (attempt === undefined) {
      continue;
    }
    const issue = issueFromAttempt(attempt);
    if (issue !== undefined) {
      return { attempt, issue };
    }
  }
  return { attempt: latestAttempt(run), issue: undefined };
}

function durationMs(run: RunStatus): number | undefined {
  const start = run.startedAt ?? run.createdAt;
  const end = run.finishedAt ?? run.updatedAt;
  if (start === undefined || end === undefined) {
    return undefined;
  }
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs < startMs) {
    return undefined;
  }
  return endMs - startMs;
}

function aggregateMetrics(run: RunStatus): RunTriageMetrics | undefined {
  const reported = run.attempts.flatMap((attempt) =>
    attempt.metrics === undefined ? [] : [attempt.metrics],
  );
  if (reported.length === 0) {
    return undefined;
  }
  return {
    modelCalls: reported.reduce((total, value) => total + value.modelCalls, 0),
    totalTokens: reported.reduce(
      (total, value) => total + value.totalTokens,
      0,
    ),
    toolCalls: reported.reduce((total, value) => total + value.toolCalls, 0),
    errorCount: reported.reduce((total, value) => total + value.errorCount, 0),
    incomplete:
      reported.length !== run.attempts.length ||
      reported.some((value) => !value.reportsComplete || value.truncated),
  };
}

export function deriveRunTriage(run: RunStatus): RunTriage {
  const context = issueContext(run);
  const duration = durationMs(run);
  const metrics = aggregateMetrics(run);
  return {
    ...(context.attempt === undefined
      ? {}
      : {
          stage: context.attempt.stage,
          stageExecutionId: context.attempt.stageExecutionId,
        }),
    attemptCount: run.attempts.length,
    ...(duration === undefined ? {} : { durationMs: duration }),
    outputCount: Object.keys(run.outputs).length,
    ...(context.issue === undefined ? {} : { issue: context.issue }),
    ...(metrics === undefined ? {} : { metrics }),
  };
}

export function formatRunDuration(value: number | undefined): string {
  if (value === undefined) {
    return "—";
  }
  const seconds = Math.floor(value / 1000);
  if (seconds < 1) {
    return "<1s";
  }
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ${minutes % 60}m`;
  }
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}
