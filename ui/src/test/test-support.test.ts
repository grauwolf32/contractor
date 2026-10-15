import { describe, expect, it } from "vitest";

import {
  audit,
  conforming,
  finding,
  project,
  report,
  review,
} from "./builders";
import {
  failure,
  fakeServer,
  invalidJSON,
  json,
  operationOf,
  page,
} from "./fake-server";
import { testAPI } from "./render";

describe("fake Server", () => {
  it("matches requests to public operations", () => {
    expect(operationOf("GET", "/v1/unknown")).toBeUndefined();
    expect(operationOf("PATCH", "/v1/audits/audit_1")).toBeUndefined();
    expect(operationOf("GET", "/v1/audits/audit_1")).toEqual({
      operation: "GET /v1/audits/{auditId}",
      params: { auditId: "audit_1" },
    });
    expect(operationOf("GET", "/v1/audit-profiles")?.operation).toBe(
      "GET /v1/audit-profiles",
    );
  });

  it("routes by operation, decodes parameters and records requests", async () => {
    const server = fakeServer({
      "GET /v1/projects/{projectId}": ({ params }) =>
        json(project(params.projectId ?? ""), 200, { ETag: '"1"' }),
    });
    const { api } = testAPI(server);
    const result = await api.request((client) =>
      client.GET("/v1/projects/{projectId}", {
        params: { path: { projectId: "shop.v2" } },
      }),
    );
    expect(result.data?.projectId).toBe("shop.v2");
    expect(server.sent("GET", "/projects/shop.v2")).toHaveLength(1);
  });

  it("answers unrouted requests from the fallback, then with 404", async () => {
    const server = fakeServer({}, (_, url) =>
      url.pathname === "/v1/projects" ? json(page([])) : undefined,
    );
    const { api } = testAPI(server);
    await expect(
      api.request((client) => client.GET("/v1/projects")),
    ).resolves.toMatchObject({ data: { items: [] } });
    const missing = await api.request((client) =>
      client.GET("/v1/projects/{projectId}", {
        params: { path: { projectId: "none" } },
      }),
    );
    expect(missing.response.status).toBe(404);
  });

  it.fails(
    "fails the test that serves a body breaking the contract",
    async () => {
      const server = fakeServer({
        "GET /v1/projects/{projectId}": () => json({ projectId: "x" }),
      });
      const { api } = testAPI(server);
      await api
        .request((client) =>
          client.GET("/v1/projects/{projectId}", {
            params: { path: { projectId: "x" } },
          }),
        )
        .catch(() => undefined);
    },
  );

  it("lets a probe send a body that breaks the contract", async () => {
    const server = fakeServer({
      "GET /v1/projects/{projectId}": () => invalidJSON({ projectId: "x" }),
    });
    const { api } = testAPI(server);
    await expect(
      api.request((client) =>
        client.GET("/v1/projects/{projectId}", {
          params: { path: { projectId: "x" } },
        }),
      ),
    ).rejects.toMatchObject({ code: "invalid_api_response", status: 200 });
    expect(failure(503).status).toBe(503);
  });
});

describe("builders", () => {
  it("build conforming fixtures and reject broken ones", () => {
    const check = audit("project_1", "audit_1", "completed");
    expect(check.phase).toBe("rounds");
    expect(audit("project_1", "audit_2", "draft").phase).toBe("not-started");
    expect(finding("audit_1", "finding_1").state).toBe("proposed");
    expect(review("audit_1", "review_1", "finding-triage").subjectKind).toBe(
      "finding",
    );
    expect(report("audit_1", "proposed").review?.kind).toBe(
      "report-acceptance",
    );
    expect(() =>
      audit("project_1", "audit_3", "draft", { inputs: {} }),
    ).toThrow("Audit fixture breaks the contract at #/inputs");
    expect(() =>
      conforming("Project", { ...project("p"), revision: "0" }),
    ).toThrow("#/revision");
  });
});
