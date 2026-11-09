import type { LLMCost, CostScope } from "../api/llm-cost";
import { conforming } from "./builders";
import {
  researchFixture,
  RESEARCH_AUDIT,
  RESEARCH_PROJECT,
} from "./audit-research-fixtures";
import { runStatus, workflowResource } from "./builders-execution";
import { failure, json, page, unavailableCostForURL } from "./responses";

/** Synthetic observations for visual and component tests, never product state. */
export function costFixture(
  scope: CostScope = "audit",
  id = "audit_research",
  overrides: Partial<LLMCost> = {},
): LLMCost {
  const amount =
    scope === "project" ? "48.76" : scope === "audit" ? "12.84" : "0.00000625";
  return conforming("LLMCostSummary", {
    scope,
    scopeId: id,
    currency: "USD",
    amount,
    calls: "86",
    pricedCalls: "86",
    reports: "6",
    unknownReports: "0",
    basis: "reported",
    coverage: "complete",
    pending: false,
    historicalGap: false,
    asOf: "2026-11-08T12:00:00Z",
    updatedAt: "2026-11-08T11:59:00Z",
    checksAmount:
      scope === "project" ? "37.7" : scope === "audit" ? amount : "0",
    standaloneAmount:
      scope === "project" ? "11.06" : scope === "run" ? amount : "0",
    activity: [
      {
        key: "worker",
        label: "Worker analysis",
        calls: "86",
        amount,
        basis: "reported",
      },
    ],
    models: [
      {
        key: "model:openrouter/demo-worker",
        label: "openrouter/demo-worker",
        calls: "86",
        amount,
        basis: "reported",
      },
    ],
    executions: [],
    ...overrides,
  });
}

export function costResearchFixture(state = "normal") {
  const research = researchFixture(state === "pending" ? "active" : "normal");
  const run = runStatus("run_1", {
    workflow: "analysis@1",
    projectId: RESEARCH_PROJECT,
  });
  let costReads = 0;
  return {
    research,
    async respond(request: Request): Promise<Response | undefined> {
      const url = new URL(request.url),
        unknown = unavailableCostForURL(request.method, url);
      if (unknown) {
        costReads++;
        if (state === "failed-refresh" && costReads > 1)
          return failure(503, "storage_transaction_conflict");
        let value = costFixture(unknown.scope, unknown.scopeId);
        if (state === "unavailable") value = unknown;
        if (state === "partial")
          Object.assign(value, {
            coverage: "partial",
            pricedCalls: "70",
            unknownReports: "2",
            historicalGap: true,
            activity: value.activity.map((row) => ({ ...row, calls: "70" })),
            models: value.models.map((row) => ({ ...row, calls: "70" })),
          });
        if (state === "pending" || state === "failed-refresh")
          Object.assign(value, { pending: true, coverage: "pending" });
        if (state === "estimate")
          Object.assign(value, {
            basis: "estimated",
            activity: value.activity.map((row) => ({
              ...row,
              basis: "estimated",
            })),
            models: value.models.map((row) => ({ ...row, basis: "estimated" })),
          });
        if (state === "zero")
          Object.assign(value, {
            amount: "0",
            checksAmount: "0",
            standaloneAmount: "0",
            calls: "0",
            pricedCalls: "0",
            basis: "none",
            activity: [],
            models: [],
          });
        value.executions = url.searchParams
          .getAll("executionId")
          .map((executionId) => ({
            executionId,
            runId: "run_1",
            amount: state === "unavailable" ? null : "0.00000625",
            basis: state === "unavailable" ? "none" : "reported",
            coverage: state === "unavailable" ? "unavailable" : "complete",
            pending: false,
          }));
        return json(conforming("LLMCostSummary", value));
      }
      if (url.pathname === "/v1/runs/run_1") return json(run);
      if (url.pathname === "/v1/workflows/analysis/versions/1")
        return json(workflowResource("analysis", "1"));
      if (url.pathname === `/v1/projects/${RESEARCH_PROJECT}/findings`) {
        const response = await research.respond(
          new Request(url.origin + `/v1/audits/${RESEARCH_AUDIT}/findings`),
        );
        const data = await response!.json();
        return json({
          items: data.items,
          page: data.page,
          projectRevision: data.projectRevision,
        });
      }
      if (url.pathname === `/v1/projects/${RESEARCH_PROJECT}/runs`)
        return json(page([]));
      if (url.pathname === "/v1/projects")
        return json(page([research.project]));
      if (url.pathname === "/v1/audits") return json(page([research.audit]));
      if (url.pathname === `/v1/audits/${RESEARCH_AUDIT}`)
        return json(research.audit, 200, {
          ETag: `"${research.audit.revision}"`,
        });
      return research.respond(request);
    },
  };
}
