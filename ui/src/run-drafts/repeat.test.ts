import { describe, expect, it } from "vitest";

import type { RunRepeatDraftResponse } from "../api/runs";
import { auditDestination, prepareRepeatDraft } from "./repeat";

const response: RunRepeatDraftResponse = {
  sourceRunId: "run-source",
  authority: "ordinary",
  workflow: { name: "inspect", version: "2" },
  projectId: "project-one",
  notices: [
    {
      code: "runtime_binding_changed",
      severity: "warning",
      field: "runtimeLabels.debug",
      message: "Debug now resolves differently.",
    },
  ],
  draft: {
    parameters: { objective: "Inspect exact source" },
    runtimeLabels: ["debug"],
    labels: { "eval.id": "sample-1", purpose: "eval" },
    executionConfig: {
      status: "available",
      value: {
        planner: { modelPolicy: "planner@2", credential: null },
        workers: { llmGateway: "local@1", credential: "worker-budget" },
        stages: {
          inspect: {
            agents: { reviewer: { modelPolicy: "reviewer@3" } },
          },
        },
      },
    },
    inputs: {
      source: {
        status: "available",
        sourceScope: "project",
        artifact: {
          namespace: "sources",
          name: "service",
          revision: "source-r1",
        },
        metadata: {
          artifact: {
            namespace: "sources",
            name: "service",
            revision: "source-r1",
          },
          mediaType: "application/zip",
          size: 42,
          current: false,
          frozen: false,
          createdAt: "2026-09-07T10:00:00Z",
        },
      },
      baseline: {
        status: "unavailable",
        artifact: {
          namespace: "reports",
          name: "baseline",
          revision: "gone-r1",
        },
        code: "input_source_unavailable",
        message: "The exact source is gone.",
      },
    },
  },
};

describe("repeat Run drafts", () => {
  it("retains only available exact sources and caller-controlled settings", () => {
    const prepared = prepareRepeatDraft(response);
    expect(prepared).toBeDefined();
    expect(prepared?.identity).toEqual({
      workflowName: "inspect",
      workflowVersion: "2",
      projectId: "project-one",
    });
    expect(prepared?.destination).toBe(
      "/projects/project-one/workflows/inspect/2/run",
    );
    expect(prepared?.state.parameters).toEqual({
      objective: "Inspect exact source",
    });
    expect(prepared?.state.artifactSelections).toEqual({
      source: "sources/service@source-r1",
    });
    expect(prepared?.state.artifactReviews).toEqual({});
    expect(prepared?.state.runtimeLabels).toEqual(["debug"]);
    expect(prepared?.state.overrides).toMatchObject({
      planner: {
        modelPolicy: "planner@2",
        credential: "__none__",
      },
      workers: {
        llmGateway: "local@1",
        credential: "worker-budget",
      },
      stages: response.draft?.executionConfig.value?.stages,
    });
    expect(prepared?.state.repeat).toMatchObject({
      sourceRunId: "run-source",
      reviewed: false,
    });
    expect(
      prepared?.state.repeat?.notices.some(
        (notice) => notice.code === "stage_execution_overrides_retained",
      ),
    ).toBe(true);
  });

  it("routes Audit-managed Runs to their owning Audit without a draft", () => {
    const audit: RunRepeatDraftResponse = {
      sourceRunId: "run-audit",
      authority: "audit-managed",
      workflow: { name: "audit-check", version: "1" },
      projectId: "project-one",
      auditId: "audit-one",
      notices: [],
    };
    expect(prepareRepeatDraft(audit)).toBeUndefined();
    expect(auditDestination(audit)).toBe(
      "/projects/project-one/audits/audit-one",
    );
  });
});
