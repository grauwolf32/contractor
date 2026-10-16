// Workflow, Run, configuration and Operations fixtures, checked against the
// generated schemas when they are built (see builders.ts).
import type { components } from "../api/generated/public";
import { conforming, DIGEST, NOW } from "./builders";

type Schemas = components["schemas"];

/** One Stage that succeeds into the end of its Workflow. */
export function stageContract(
  overrides: Partial<Schemas["WorkflowStageContract"]> = {},
): Schemas["WorkflowStageContract"] {
  return conforming("WorkflowStageContract", {
    objective: "Analyze the materials.",
    instructions: { ref: "instructions/analyze.md", digest: DIGEST },
    planner: { plannerId: "streamline", version: "1" },
    agents: {
      analyst: {
        template: { templateId: "analyst", version: "1", digest: DIGEST },
        namespace: "analysis",
        skills: [],
      },
    },
    executionConfig: { agents: { analyst: {} } },
    contextArtifacts: {},
    resultArtifacts: {},
    workflowOutputs: {},
    on: {
      succeeded: { kind: "succeed" },
      failed: { kind: "fail" },
      interrupted: { kind: "fail" },
    },
    ...overrides,
  });
}

/** An exact Workflow version whose entry Stage is `execute`. */
export function workflowResource(
  name: string,
  version = "1",
  overrides: Partial<Schemas["WorkflowResource"]> = {},
): Schemas["WorkflowResource"] {
  return conforming("WorkflowResource", {
    ref: { name, version },
    entryStage: "execute",
    parameters: {},
    inputs: {},
    outputs: {},
    stages: { execute: stageContract() },
    ...overrides,
  });
}

/** The pins of a Run that selected no Runtime labels. */
export function runRuntimeConfiguration(): Schemas["RunRuntimeConfiguration"] {
  return conforming("RunRuntimeConfiguration", {
    default: {
      label: "default",
      bindingRevision: "1",
      config: {
        name: "contractor-empty",
        version: "1",
        digest: `sha256:${"0".repeat(64)}`,
      },
    },
    labels: [],
  });
}

/** One Stage attempt of `stage`. */
export function stageAttempt(
  stageExecutionId: string,
  overrides: Partial<Schemas["StageAttempt"]> = {},
): Schemas["StageAttempt"] {
  return conforming("StageAttempt", {
    stageExecutionId,
    stage: "execute",
    attempt: 1,
    executionConfig: { variant: "base", agents: { analyst: {} } },
    state: "succeeded",
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  });
}

/** A Run detail of `workflow`; attempts and outputs come from overrides. */
export function runStatus(
  runId: string,
  overrides: Partial<Schemas["RunStatus"]> = {},
): Schemas["RunStatus"] {
  return conforming("RunStatus", {
    runId,
    workflow: "analysis@1",
    state: "succeeded",
    deletable: true,
    runtimeLabels: [],
    labels: {},
    runtimeConfiguration: runRuntimeConfiguration(),
    attempts: [],
    transitions: [],
    outputs: {},
    outputPublications: [],
    createdAt: NOW,
    updatedAt: NOW,
    ...overrides,
  });
}

/** The catalog list item of an exact Workflow version: it has no Stages. */
export function workflowSummary(
  resource: Schemas["WorkflowResource"],
): Schemas["WorkflowSummary"] {
  const summary: Partial<Schemas["WorkflowResource"]> = { ...resource };
  delete summary.stages;
  return conforming("WorkflowSummary", summary as Schemas["WorkflowSummary"]);
}

/** An empty Operations snapshot at `revision`. */
export function operationsSnapshot(
  revision = "7",
  overrides: Partial<Schemas["OperationsSnapshot"]> = {},
): Schemas["OperationsSnapshot"] {
  return conforming("OperationsSnapshot", {
    cursor: { generation: "operations-generation-1", revision },
    runtimeAgents: [],
    allocations: [],
    ...overrides,
  });
}
