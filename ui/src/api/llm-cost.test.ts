import { describe, expect, it } from "vitest";
import { getLLMCost } from "./llm-cost";
import { costFixture } from "../test/llm-cost-fixtures";
import { fakeServer } from "../test/fake-server";
import { json } from "../test/responses";
import { testAPI } from "../test/render";

describe("LLM cost API", () => {
  it("keeps exact monetary strings and uses one bounded repeated-ID read", async () => {
    const server = fakeServer({
      "GET /v1/audits/{auditId}/llm-cost": ({ url }) => {
        expect(url.searchParams.getAll("executionId")).toEqual([
          "execution_1",
          "execution_2",
        ]);
        return json(
          costFixture("audit", "audit_research", {
            executions: ["execution_1", "execution_2"].map((executionId) => ({
              executionId,
              runId: null,
              amount: "0.00000625",
              basis: "reported",
              coverage: "complete",
              pending: false,
            })),
          }),
        );
      },
    });
    const { api } = testAPI(server);
    const got = await getLLMCost(api, "audit", "audit_research", undefined, [
      "execution_1",
      "execution_2",
    ]);
    expect(got.executions[0]?.amount).toBe("0.00000625");
    expect(server.sent("GET", "/llm-cost")).toHaveLength(1);
  });
  it.each([
    ["foreign scope", { scopeId: "foreign" }],
    ["double counted child", { standaloneAmount: "12.84" }],
    ["impossible coverage", { pricedCalls: "87" }],
    ["unknown shown as zero", { coverage: "unavailable" }],
    ["inconsistent breakdown", { models: [] }],
  ])("rejects %s", async (_, change) => {
    const { api } = testAPI(async () => json({ ...costFixture(), ...change }));
    await expect(
      getLLMCost(api, "audit", "audit_research"),
    ).rejects.toMatchObject({
      detail: "Server returned an inconsistent LLM cost summary",
    });
  });
});
