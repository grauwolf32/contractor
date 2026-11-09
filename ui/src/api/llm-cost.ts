import { usdUnits } from "../app/format";
import type { PublicAPI } from "./client";
import { invalidAPIResponse, requireData } from "./error";
import type { components } from "./generated/public";

export type LLMCost = components["schemas"]["LLMCostSummary"];
export type CostScope = LLMCost["scope"];

export async function getLLMCost(
  api: PublicAPI,
  scope: CostScope,
  id: string,
  signal?: AbortSignal,
  executionIds: readonly string[] = [],
): Promise<LLMCost> {
  const result = await api.request((client) => {
    if (scope === "run")
      return client.GET("/v1/runs/{runId}/llm-cost", {
        ...(signal === undefined ? {} : { signal }),
        params: { path: { runId: id } },
      });
    if (scope === "project")
      return client.GET("/v1/projects/{projectId}/llm-cost", {
        ...(signal === undefined ? {} : { signal }),
        params: { path: { projectId: id } },
      });
    return client.GET("/v1/audits/{auditId}/llm-cost", {
      ...(signal === undefined ? {} : { signal }),
      params: {
        path: { auditId: id },
        query: executionIds.length ? { executionId: [...executionIds] } : {},
      },
      querySerializer: { array: { style: "form", explode: true } },
    });
  });
  const value = requireData(result);
  // Generated schemas check individual fields. The prepared aggregate must
  // also agree with itself and the requested scope; no client-side pricing.
  if (
    value.scope !== scope ||
    value.scopeId !== id ||
    BigInt(value.pricedCalls) > BigInt(value.calls) ||
    BigInt(value.unknownReports) > BigInt(value.reports) ||
    (value.amount === null) !== (value.coverage === "unavailable") ||
    (value.amount !== null &&
      (value.checksAmount === null ||
        value.standaloneAmount === null ||
        usdUnits(value.checksAmount) + usdUnits(value.standaloneAmount) !==
          usdUnits(value.amount) ||
        value.models.reduce((n, row) => n + usdUnits(row.amount), 0n) !==
          usdUnits(value.amount) ||
        value.activity.reduce((n, row) => n + usdUnits(row.amount), 0n) !==
          usdUnits(value.amount))) ||
    value.executions.length !== executionIds.length ||
    new Set(value.executions.map((row) => row.executionId)).size !==
      value.executions.length ||
    value.executions.some((row) => !executionIds.includes(row.executionId))
  )
    throw invalidAPIResponse(
      result.response.status,
      "Server returned an inconsistent LLM cost summary",
    );
  return value;
}
