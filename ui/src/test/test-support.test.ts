import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { schemaViolation } from "../api/contract";
import {
  audit,
  conforming,
  finding,
  project,
  report,
  review,
} from "./builders";
import { createEvalFixture } from "./evals-fixture";
import { fakeServer, invalidJSON } from "./fake-server";
import { bytes, failure, json, operationOf, page } from "./responses";
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

  it("serves artifact content as raw bytes, also when it is JSON", async () => {
    const server = fakeServer({
      "GET /v1/runs/{runId}/artifacts/{namespace}/{name}": () =>
        bytes('{"anything": "goes"}', "application/json"),
    });
    const response = await server.fetch(
      "http://127.0.0.1:8080/v1/runs/run_1/artifacts/outputs/trace",
    );
    expect(response.headers.get("Content-Length")).toBe("20");
    expect(await response.text()).toBe('{"anything": "goes"}');
    expect(
      server.handles("GET /v1/runs/{runId}/artifacts/{namespace}/{name}"),
    ).toBe(true);
    expect(server.handles("GET /v1/projects")).toBe(false);
  });
});

describe("browser-safe helpers", () => {
  // The Playwright specs import these; a value import of vitest (directly or
  // through fake-server.ts) would load the unit test runner into the browser
  // run.
  it.each([
    "responses.ts",
    "builders.ts",
    "builders-execution.ts",
    "owner-lists.ts",
    "evals-fixture.ts",
    "audit-presets-fixture.ts",
  ])("%s imports no test runner", (file) => {
    const source = readFileSync(new URL(file, import.meta.url), "utf8");
    const runtimeImports = [
      ...source.matchAll(/^import (?!type )[^;]*?from "([^"]+)";/gms),
    ].map((match) => match[1]);
    expect(runtimeImports).not.toContain("vitest");
    expect(runtimeImports).not.toContain("./fake-server");
    expect(runtimeImports).not.toContain("./render");
  });
});

describe("Eval fixture", () => {
  it("answers errors with the contract's error envelope", async () => {
    const fixture = createEvalFixture();
    const refused = await fixture.fetch(
      new Request("http://127.0.0.1:8080/v1/eval-experiments", {
        method: "POST",
        body: "{}",
      }),
    );
    expect(refused.status).toBe(403);
    expect(schemaViolation("Error", await refused.json())).toBeUndefined();
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
