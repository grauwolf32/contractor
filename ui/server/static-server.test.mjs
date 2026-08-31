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

async function fixture(t) {
  const distDir = await mkdtemp(join(tmpdir(), "contractor-ui-test-"));
  await mkdir(join(distDir, "assets"));
  await writeFile(
    join(distDir, "index.html"),
    "<!doctype html><title>UI</title>",
  );
  await writeFile(join(distDir, "assets", "app-12345678.js"), "export {};");
  const server = await createStaticServer({ distDir, runtimeConfig });
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

function rawRequest(origin, path, method = "GET") {
  const url = new URL(origin);
  return new Promise((resolve, reject) => {
    const request = httpRequest(
      {
        host: url.hostname,
        port: url.port,
        method,
        path,
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

test("known client routes get no-store index and a derived CSP", async (t) => {
  const { origin } = await fixture(t);
  for (const route of [
    "/",
    "/login",
    "/workflows",
    "/artifacts",
    "/artifacts/projects/source",
    "/runs",
    "/operations",
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
    "/unknown",
    "/manifest.json",
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
