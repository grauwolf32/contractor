import { constants as fsConstants } from "node:fs";
import { open, realpath, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";

const CLIENT_ROUTES = new Set([
  "/",
  "/login",
  "/projects",
  "/evals",
  "/queue",
  "/workflows",
  "/artifacts",
  "/runs",
  "/runs/configuration",
  "/skills",
  "/catalog",
  "/catalog/workflows",
  "/catalog/agents",
  "/catalog/skills",
  "/operations",
  "/operations/performance",
  "/operations/allocations/completed",
]);
const CLIENT_ROUTE_PATTERNS = [
  /^\/catalog\/(?:workflows|agents)\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/,
  /^\/projects\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/,
  /^\/evals\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/,
  /^\/evals\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/artifacts\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/,
  /^\/projects\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/artifacts\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/,
  /^\/projects\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/audits$/,
  /^\/projects\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/audits\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/,
  /^\/projects\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/audits\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/(?:overview|coverage|findings|checks|reviews|runs|report)$/,
  /^\/artifacts\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/,
  /^\/workflows\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/,
  /^\/runs\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/,
  /^\/runs\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\/artifacts\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/,
  /^\/operations\/(?:runtime-agents|runtime-configs|allocations|configurations|credentials|settings)$/,
  /^\/operations\/runtime-configs\/[a-z][a-z0-9_-]{0,62}\/[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/,
  /^\/runs\/configuration\/[a-z][a-z0-9_-]{0,62}\/[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/,
  /^\/operations\/configurations\/(?:agent-templates|execution-configs|model-policies|llm-gateways)\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/,
  /^\/operations\/credentials\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/,
];
const API_PREFIXES = ["/v1", "/api", "/private"];
const HASHED_ASSET = /-[A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+$/;
const LIKEC4_WORKER_ASSET = /^\/assets\/likec4\.worker-[A-Za-z0-9_-]{8,}\.js$/;
const ASSET_PATH = /^\/assets\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9._-]+$/;
const MAXIMUM_REQUEST_TARGET_BYTES = 4096;

const CONTENT_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".png", "image/png"],
  [".svg", "image/svg+xml"],
  [".woff2", "font/woff2"],
]);

function websocketOrigin(apiBaseUrl) {
  const parsed = new URL(apiBaseUrl);
  parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  return parsed.origin;
}

function isLoopbackIPLiteral(hostname) {
  if (hostname === "[::1]") {
    return true;
  }
  const octets = hostname.split(".");
  return (
    octets.length === 4 &&
    octets[0] === "127" &&
    octets.every((octet) => /^(?:0|[1-9][0-9]{0,2})$/.test(octet)) &&
    octets.every((octet) => Number(octet) <= 255)
  );
}

function apiBaseUrlForRequest(request, configured) {
  const authority = request.headers.host;
  if (
    typeof authority !== "string" ||
    authority.length === 0 ||
    authority.length > 255
  ) {
    return configured;
  }
  let requestOrigin;
  try {
    requestOrigin = new URL(`http://${authority}`);
  } catch {
    return configured;
  }
  if (
    requestOrigin.username !== "" ||
    requestOrigin.password !== "" ||
    !isLoopbackIPLiteral(requestOrigin.hostname)
  ) {
    return configured;
  }
  const api = new URL(configured);
  if (api.protocol !== "http:" || isLoopbackIPLiteral(api.hostname)) {
    return configured;
  }
  api.hostname = requestOrigin.hostname;
  return api.origin;
}

function securityHeaders(apiBaseUrl) {
  const connectSources = new Set([
    "'self'",
    apiBaseUrl,
    websocketOrigin(apiBaseUrl),
  ]);
  return {
    "Content-Security-Policy": [
      "default-src 'none'",
      "base-uri 'none'",
      "frame-ancestors 'none'",
      "form-action 'self'",
      "object-src 'none'",
      "script-src 'self'",
      "worker-src 'self'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data:",
      "font-src 'self'",
      `connect-src ${[...connectSources].join(" ")}`,
    ].join("; "),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy":
      "camera=(), display-capture=(), geolocation=(), microphone=(), payment=(), usb=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
  };
}

function likeC4WorkerHeaders(baseHeaders) {
  return {
    ...baseHeaders,
    // hpcc-js Graphviz generates small WASM binding functions at runtime.
    // Keep that capability isolated to this bundled Worker: Artifact code has
    // no DOM access and the Worker cannot make network requests.
    "Content-Security-Policy": [
      "default-src 'none'",
      "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'",
      "connect-src 'none'",
    ].join("; "),
  };
}

function send(request, response, statusCode, headers, body = Buffer.alloc(0)) {
  response.writeHead(statusCode, {
    ...headers,
    "Content-Length": String(body.byteLength),
  });
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  response.end(body);
}

function sendError(request, response, statusCode, message, baseHeaders) {
  send(
    request,
    response,
    statusCode,
    {
      ...baseHeaders,
      "Cache-Control": "no-store",
      "Content-Type": "text/plain; charset=utf-8",
    },
    Buffer.from(`${message}\n`, "utf8"),
  );
}

function requestPath(requestTarget) {
  if (
    typeof requestTarget !== "string" ||
    requestTarget.length === 0 ||
    !requestTarget.startsWith("/") ||
    requestTarget.startsWith("//") ||
    Buffer.byteLength(requestTarget) > MAXIMUM_REQUEST_TARGET_BYTES
  ) {
    throw new URIError("invalid request target");
  }
  const rawPath = requestTarget.split(/[?#]/u, 1)[0];
  if (/%(?:2f|5c)/iu.test(rawPath)) {
    throw new URIError("encoded path separator");
  }
  const decodedPath = decodeURIComponent(rawPath);
  if (
    decodedPath.includes("\\") ||
    decodedPath.includes("\0") ||
    decodedPath
      .split("/")
      .some((segment) => segment === "." || segment === "..")
  ) {
    throw new URIError("unsafe path");
  }
  return new URL(requestTarget, "http://contractor-ui.invalid").pathname;
}

async function readRegularFile(path) {
  const handle = await open(
    path,
    fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
  );
  try {
    const metadata = await handle.stat();
    if (!metadata.isFile()) {
      return null;
    }
    return await handle.readFile();
  } finally {
    await handle.close();
  }
}

function isAPILookingPath(path) {
  return API_PREFIXES.some(
    (prefix) => path === prefix || path.startsWith(`${prefix}/`),
  );
}

export async function createStaticServer({ distDir, runtimeConfig }) {
  const distRoot = await realpath(distDir);
  const indexPath = resolve(distRoot, "index.html");
  const indexMetadata = await stat(indexPath);
  if (!indexMetadata.isFile()) {
    throw new Error("UI index is not a regular file");
  }
  const index = await readRegularFile(indexPath);
  if (index === null) {
    throw new Error("UI index is not a regular file");
  }
  const healthBody = Buffer.from(
    `${JSON.stringify({ status: "ok", uiVersion: runtimeConfig.uiVersion })}\n`,
    "utf8",
  );

  return createServer((request, response) => {
    const requestAPIBaseURL = apiBaseUrlForRequest(
      request,
      runtimeConfig.apiBaseUrl,
    );
    const baseHeaders = securityHeaders(requestAPIBaseURL);
    void (async () => {
      if (request.method !== "GET" && request.method !== "HEAD") {
        sendError(request, response, 405, "method not allowed", {
          ...baseHeaders,
          Allow: "GET, HEAD",
        });
        return;
      }
      let path;
      try {
        path = requestPath(request.url);
      } catch {
        sendError(
          request,
          response,
          400,
          "invalid request target",
          baseHeaders,
        );
        return;
      }
      if (path === "/healthz") {
        send(
          request,
          response,
          200,
          {
            ...baseHeaders,
            "Cache-Control": "no-store",
            "Content-Type": "application/json; charset=utf-8",
          },
          healthBody,
        );
        return;
      }
      if (path === "/runtime-config.json") {
        const runtimeConfigBody = Buffer.from(
          `${JSON.stringify({
            ...runtimeConfig,
            apiBaseUrl: requestAPIBaseURL,
          })}\n`,
          "utf8",
        );
        send(
          request,
          response,
          200,
          {
            ...baseHeaders,
            "Cache-Control": "no-store",
            "Content-Type": "application/json; charset=utf-8",
          },
          runtimeConfigBody,
        );
        return;
      }
      if (
        CLIENT_ROUTES.has(path) ||
        CLIENT_ROUTE_PATTERNS.some((pattern) => pattern.test(path))
      ) {
        send(
          request,
          response,
          200,
          {
            ...baseHeaders,
            "Cache-Control": "no-store",
            "Content-Type": "text/html; charset=utf-8",
          },
          index,
        );
        return;
      }
      if (path.startsWith("/assets/")) {
        if (!ASSET_PATH.test(path)) {
          sendError(request, response, 404, "not found", baseHeaders);
          return;
        }
        const assetPath = resolve(distRoot, `.${path}`);
        if (!assetPath.startsWith(`${distRoot}${sep}`)) {
          sendError(request, response, 404, "not found", baseHeaders);
          return;
        }
        let asset;
        try {
          asset = await readRegularFile(assetPath);
        } catch {
          asset = null;
        }
        if (asset === null) {
          sendError(request, response, 404, "not found", baseHeaders);
          return;
        }
        send(
          request,
          response,
          200,
          {
            ...(LIKEC4_WORKER_ASSET.test(path)
              ? likeC4WorkerHeaders(baseHeaders)
              : baseHeaders),
            "Cache-Control": HASHED_ASSET.test(path)
              ? "public, max-age=31536000, immutable"
              : "no-cache",
            "Content-Type":
              CONTENT_TYPES.get(extname(path).toLowerCase()) ??
              "application/octet-stream",
          },
          asset,
        );
        return;
      }
      if (isAPILookingPath(path) || extname(path) !== "") {
        sendError(request, response, 404, "not found", baseHeaders);
        return;
      }
      sendError(request, response, 404, "not found", baseHeaders);
    })().catch(() => {
      if (!response.headersSent) {
        sendError(request, response, 500, "internal server error", baseHeaders);
      } else {
        response.destroy();
      }
    });
  });
}
