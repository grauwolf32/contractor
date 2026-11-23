import { describe, expect, it } from "vitest";

import { fakeServer, invalidJSON } from "../test/fake-server";
import { testAPI } from "../test/render";
import { bytes, failure, json } from "../test/responses";
import { testSharedConfig } from "../test/shared-config";
import { downloadArtifact, writeArtifact } from "./artifacts";
import { ARTIFACT_STORAGE_TIMESTAMPS } from "../test/builders";
import { readAuditArtifact } from "./audit-artifacts";
import { canBrowseArchive } from "./artifact-archive";
import { getSharedConfig } from "./shared-config";
import { PUBLIC_PAGED_OPERATIONS } from "./generated/response-schemas";

describe("Server shared policy", () => {
  it("restarts a cancelled bootstrap without discarding a completed snapshot", async () => {
    let began!: () => void;
    const started = new Promise<void>((resolve) => {
      began = resolve;
    });
    let calls = 0;
    const server = fakeServer({
      "GET /v1/config": ({ request }) => {
        calls++;
        if (calls !== 1) return json(testSharedConfig(17));
        return new Promise<Response>((_, reject) => {
          request.signal.addEventListener("abort", () =>
            reject(new DOMException("Cancelled", "AbortError")),
          );
          began();
        });
      },
    });
    const { api } = testAPI(server);
    const firstController = new AbortController();
    const first = getSharedConfig(api, firstController.signal);
    void first.catch(() => undefined);
    await started;
    firstController.abort();
    const secondController = new AbortController();
    const snapshot = await getSharedConfig(api, secondController.signal);
    await expect(first).rejects.toBeDefined();
    secondController.abort();
    expect(await getSharedConfig(api)).toBe(snapshot);
    expect(calls).toBe(2);
  });

  it("verifies and freezes one snapshot per API client, sharing concurrent reads", async () => {
    const server = fakeServer({
      "GET /v1/config": () => json(testSharedConfig(17)),
    });
    const { api } = testAPI(server);
    const [first, second] = await Promise.all([
      getSharedConfig(api),
      getSharedConfig(api),
    ]);
    expect(first).toBe(second);
    expect(Object.isFrozen(first)).toBe(true);
    expect(Object.isFrozen(first.limits)).toBe(true);
    server.route("GET /v1/config", () => json(testSharedConfig(18)));
    expect((await getSharedConfig(api)).limits.artifactPayloadMaxBytes).toBe(
      17,
    );
    expect(server.sent("GET", "/v1/config")).toHaveLength(1);
    expect(
      (await getSharedConfig(testAPI(server).api)).limits
        .artifactPayloadMaxBytes,
    ).toBe(18);
  });

  it.each([
    { ...testSharedConfig(17), version: 2 },
    { ...testSharedConfig(17), digest: `sha256:${"0".repeat(64)}` },
    { ...testSharedConfig(17), extra: "not allowed" },
    { ...testSharedConfig(17), limits: { artifactPayloadMaxBytes: 0 } },
  ])("rejects invalid policy without caching it: %j", async (body) => {
    const server = fakeServer({ "GET /v1/config": () => invalidJSON(body) });
    const { api } = testAPI(server);
    await expect(getSharedConfig(api)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    server.route("GET /v1/config", () => json(testSharedConfig(17)));
    expect((await getSharedConfig(api)).limits.artifactPayloadMaxBytes).toBe(
      17,
    );
    expect(server.sent("GET", "/v1/config")).toHaveLength(2);
  });

  it("does not substitute a local default for a failed read", async () => {
    const server = fakeServer({
      "GET /v1/config": () => failure(503, "server_standby", "Try later"),
    });
    const { api } = testAPI(server);
    await expect(getSharedConfig(api)).rejects.toMatchObject({
      status: 503,
      retryable: true,
    });
  });

  it("enforces fetched upload, ordinary download, Audit download and archive bounds", async () => {
    const server = fakeServer({
      "GET /v1/config": () => json(testSharedConfig(17)),
      "GET /v1/artifacts/{namespace}/{name}": () =>
        bytes("x".repeat(18), "application/zip"),
      "GET /v1/audits/{auditId}/artifacts": () =>
        bytes("x".repeat(18), "application/zip"),
    });
    const { api } = testAPI(server);
    await expect(
      writeArtifact(api, {
        namespace: "inputs",
        name: "report",
        mediaType: "text/plain",
        payload: new Blob(["x".repeat(18)]),
      }),
    ).rejects.toThrow("Server upload limit");
    expect(server.sent("PUT", "/v1/artifacts/inputs/report")).toHaveLength(0);
    const metadata = {
      artifact: { namespace: "inputs", name: "report", revision: "r1" },
      mediaType: "application/zip",
      size: 18,
      current: true,
      frozen: false,
      canUploadNewVersion: true,
      createdAt: "2026-10-31T00:00:00Z",
      storageTimestamps: ARTIFACT_STORAGE_TIMESTAMPS,
    };
    await expect(downloadArtifact(api, metadata)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    await expect(
      readAuditArtifact(
        api,
        "audit-one",
        "report",
        "r1",
        undefined,
        new AbortController().signal,
      ),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
    expect(
      canBrowseArchive(
        metadata,
        (await getSharedConfig(api)).limits.artifactPayloadMaxBytes,
      ),
    ).toBe(false);
    expect(server.sent("GET", "/v1/config")).toHaveLength(1);
  });
});

describe("shared pagination policy", () => {
  it("keeps specialized Eval and allocation-history policies independent", () => {
    expect(PUBLIC_PAGED_OPERATIONS).toContain("GET /v1/projects");
    expect(PUBLIC_PAGED_OPERATIONS).not.toContain("GET /v1/evals");
    expect(PUBLIC_PAGED_OPERATIONS).not.toContain(
      "GET /v1/runs/{runId}/allocations",
    );
  });
  it("caps preferred sizes and keeps cursors and filters on every common list", async () => {
    const server = fakeServer({
      "GET /v1/config": () =>
        json(testSharedConfig(17, { publicPageMaxItems: 2 })),
      "GET /v1/projects": () => json({ items: [], page: { hasMore: false } }),
    });
    const { api } = testAPI(server);
    await api.request((client) =>
      client.GET("/v1/projects", {
        params: { query: { limit: 50, cursor: "next" } },
      }),
    );
    await api.request((client) => client.GET("/v1/projects"));
    const requests = server.sent("GET", "/v1/projects");
    expect(requests).toHaveLength(2);
    expect(new URL(requests[0]!.url).searchParams.get("limit")).toBe("2");
    expect(new URL(requests[0]!.url).searchParams.get("cursor")).toBe("next");
    expect(new URL(requests[1]!.url).searchParams.get("limit")).toBe("2");
    expect(server.sent("GET", "/v1/config")).toHaveLength(1);
  });
});
