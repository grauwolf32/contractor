import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { request as httpRequest } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createStaticServer } from "./static-server.mjs";

const runtimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

async function fixture(t, configuredRuntimeConfig = runtimeConfig) {
  const distDir = await mkdtemp(join(tmpdir(), "contractor-ui-test-"));
  await mkdir(join(distDir, "assets"));
  await writeFile(
    join(distDir, "index.html"),
    "<!doctype html><title>UI</title>",
  );
  await writeFile(join(distDir, "assets", "app-12345678.js"), "export {};");
  await writeFile(
    join(distDir, "assets", "likec4.worker-12345678.js"),
    "export {};",
  );
  const server = await createStaticServer({
    distDir,
    runtimeConfig: configuredRuntimeConfig,
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert(address && typeof address === "object");
  const origin = `http://127.0.0.1:${address.port}`;
  t.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    await rm(distDir, { recursive: true, force: true });
  });
  return { origin };
}

function rawRequest(origin, path, method = "GET", headers = {}) {
  const url = new URL(origin);
  return new Promise((resolve, reject) => {
    const request = httpRequest(
      {
        host: url.hostname,
        port: url.port,
        method,
        path,
        headers,
      },
      (response) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () =>
          resolve({
            status: response.statusCode,
            headers: response.headers,
            body: Buffer.concat(chunks).toString("utf8"),
          }),
        );
      },
    );
    request.on("error", reject);
    request.end();
  });
}

test("private-network config preserves an IP-literal loopback entrypoint", async (t) => {
  const privateConfig = {
    ...runtimeConfig,
    apiBaseUrl: "http://192.168.1.217:8080",
  };
  const { origin } = await fixture(t, privateConfig);

  const local = await rawRequest(origin, "/runtime-config.json");
  assert.equal(local.status, 200);
  assert.equal(JSON.parse(local.body).apiBaseUrl, "http://127.0.0.1:8080");

  const lan = await rawRequest(origin, "/runtime-config.json", "GET", {
    Host: "192.168.1.217:4173",
  });
  assert.equal(lan.status, 200);
  assert.deepEqual(JSON.parse(lan.body), privateConfig);

  const localIndex = await rawRequest(origin, "/");
  assert.match(
    localIndex.headers["content-security-policy"],
    /connect-src 'self' http:\/\/127\.0\.0\.1:8080 ws:\/\/127\.0\.0\.1:8080/,
  );

  const loopbackConfig = {
    ...runtimeConfig,
    apiBaseUrl: "http://127.0.0.3:8080",
  };
  const loopback = await fixture(t, loopbackConfig);
  const separateLoopbackAPI = await rawRequest(
    loopback.origin,
    "/runtime-config.json",
    "GET",
    { Host: "127.0.0.2:4173" },
  );
  assert.deepEqual(JSON.parse(separateLoopbackAPI.body), loopbackConfig);
});

test("known client routes get no-store index and a derived CSP", async (t) => {
  const { origin } = await fixture(t);
  for (const route of [
    "/",
    "/login",
    "/projects",
    "/projects/project_example",
    "/projects/project_example/artifacts/sources/service",
    "/projects/project_example/audits",
    "/projects/project_example/findings",
    "/projects/project_example/findings?audit=audit_example&severity=high",
    "/projects/project_example/audits/audit_example",
    "/projects/project_example/audits/audit_example/coverage",
    "/projects/project_example/audits/audit_example/findings",
    "/projects/project_example/audits/audit_example/reviews",
    "/evals",
    "/evals/evaluation_example",
    "/evals/evaluation_example/artifacts/sources/service",
    "/queue",
    "/workflows",
    "/workflows/openapi-from-workspace/3",
    "/artifacts",
    "/artifacts/projects/source",
    "/runs",
    "/runs/run_example",
    "/runs/run_example/artifacts/outputs/openapi",
    "/skills",
    "/catalog",
    "/catalog/workflows",
    "/catalog/workflows/openapi-from-workspace/3",
    "/catalog/agents",
    "/catalog/agents/worker/1",
    "/catalog/skills",
    "/operations",
    "/operations/runtime-agents",
    "/operations/runtime-configs",
    "/operations/runtime-configs/debug/1",
    "/runs/configuration",
    "/runs/configuration/debug/1",
    "/operations/allocations",
    "/operations/allocations/completed",
    "/operations/performance",
    "/operations/configurations",
    "/operations/configurations/model-policies/worker/1",
    "/operations/credentials",
    "/operations/credentials/worker-budget",
    "/operations/settings",
  ]) {
    const response = await fetch(`${origin}${route}`);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.match(await response.text(), /<title>UI<\/title>/);
  }
  const response = await fetch(`${origin}/runs`);
  const csp = response.headers.get("content-security-policy");
  assert.match(
    csp,
    /connect-src 'self' http:\/\/127\.0\.0\.1:8080 ws:\/\/127\.0\.0\.1:8080/,
  );
  assert.match(csp, /script-src 'self'/);
  assert.doesNotMatch(csp, /unsafe-eval/);
  assert.match(csp, /style-src 'self' 'unsafe-inline'/);
  assert.match(csp, /worker-src 'self'/);
  assert.equal(response.headers.get("x-content-type-options"), "nosniff");
  assert.equal(response.headers.get("x-frame-options"), "DENY");
});

test("runtime config, health and hashed assets have distinct cache policy", async (t) => {
  const { origin } = await fixture(t);
  const configResponse = await fetch(`${origin}/runtime-config.json`);
  assert.equal(configResponse.status, 200);
  assert.equal(configResponse.headers.get("cache-control"), "no-store");
  assert.deepEqual(await configResponse.json(), runtimeConfig);

  const healthResponse = await fetch(`${origin}/healthz`);
  assert.deepEqual(await healthResponse.json(), {
    status: "ok",
    uiVersion: "0.1.0",
  });

  const assetResponse = await fetch(`${origin}/assets/app-12345678.js`);
  assert.equal(assetResponse.status, 200);
  assert.equal(
    assetResponse.headers.get("cache-control"),
    "public, max-age=31536000, immutable",
  );
  assert.match(assetResponse.headers.get("content-type"), /javascript/);
  assert.doesNotMatch(
    assetResponse.headers.get("content-security-policy"),
    /unsafe-eval/,
  );

  const workerResponse = await fetch(
    `${origin}/assets/likec4.worker-12345678.js`,
  );
  assert.equal(workerResponse.status, 200);
  const workerCSP = workerResponse.headers.get("content-security-policy");
  assert.match(workerCSP, /script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'/);
  assert.match(workerCSP, /connect-src 'none'/);

  const headResponse = await fetch(`${origin}/operations`, { method: "HEAD" });
  assert.equal(headResponse.status, 200);
  assert.equal(await headResponse.text(), "");
  assert(Number(headResponse.headers.get("content-length")) > 0);
});

test("API-looking, missing asset, extension and unknown routes never fall back", async (t) => {
  const { origin } = await fixture(t);
  for (const path of [
    "/v1/auth/session",
    "/api/anything",
    "/private/control",
    "/assets/missing-12345678.js",
    "/settings",
    "/unknown",
    "/manifest.json",
    "/workflows/name/1/extra",
    "/catalog/unknown",
    "/catalog/agents/worker/1/extra",
    "/catalog/workflows/name/1/extra",
    "/projects/project_example/extra",
    "/projects/project_example/findings/extra",
    "/evals/evaluation_example/findings",
    "/projects/project_example/artifacts/sources/service/extra",
    "/projects/project_example/audits/audit_example/unknown",
    "/projects/project_example/audits/audit_example/coverage/extra",
    "/evals/evaluation_example/artifacts/sources/service/extra",
    "/runs/run_example/extra",
    "/runs/run_example/artifacts/outputs/openapi/extra",
    "/operations/configurations/model-policies/worker/1/extra",
    "/operations/configurations/unknown/worker/1",
    "/operations/runtime-configs/debug/1/extra",
    "/operations/runtime-configs/INVALID/1",
    "/runs/configuration/debug/1/extra",
    "/runs/configuration/INVALID/1",
    "/operations/credentials/worker-budget/extra",
  ]) {
    const response = await fetch(`${origin}${path}`);
    assert.equal(response.status, 404, path);
    assert.equal(await response.text(), "not found\n", path);
  }
  const traversal = await rawRequest(origin, "/v1/%2e%2e/runs");
  assert.equal(traversal.status, 400);
  assert.equal(traversal.body, "invalid request target\n");
  const absoluteForm = await rawRequest(origin, "http://attacker.invalid/runs");
  assert.equal(absoluteForm.status, 400);
});

test("static service rejects domain mutations without reading a body", async (t) => {
  const { origin } = await fixture(t);
  const response = await rawRequest(origin, "/runs", "POST");
  assert.equal(response.status, 405);
  assert.equal(response.headers.allow, "GET, HEAD");
  assert.equal(response.body, "method not allowed\n");
});
