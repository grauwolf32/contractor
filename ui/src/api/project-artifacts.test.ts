import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import type { ArtifactMetadata } from "./artifacts";
import { PublicAPI } from "./client";
import {
  downloadProjectArtifact,
  listProjectArtifacts,
  writeProjectArtifact,
} from "./project-artifacts";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const metadata: ArtifactMetadata = {
  artifact: {
    namespace: "sources",
    name: "service",
    revision: "revision-1",
  },
  mediaType: "application/zip",
  size: 3,
  current: true,
  frozen: false,
  createdAt: "2026-09-01T10:00:00Z",
};

function response(body: BodyInit | null, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(body, { ...options, headers });
}

describe("Project Artifact API", () => {
  it("lists ProjectScope without accepting owner or scope selectors", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response(
          JSON.stringify({ items: [metadata], page: { hasMore: false } }),
          { headers: { "content-type": "application/json" } },
        );
      }),
    );
    await listProjectArtifacts(api, {
      projectId: "project_example",
      namespace: "sources",
    });
    const url = new URL(captured!.url);
    expect(url.pathname).toBe("/v1/projects/project_example/artifacts");
    expect(url.searchParams.get("namespace")).toBe("sources");
    expect(url.searchParams.has("owner_id")).toBe(false);
    expect(url.searchParams.has("scope_kind")).toBe(false);
  });

  it("uploads bytes directly to the scope-bound endpoint with exact CAS", async () => {
    const requests: Request[] = [];
    let revision = 0;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        requests.push(request);
        revision += 1;
        return response(
          JSON.stringify({
            artifact: {
              namespace: "sources",
              name: "service",
              revision: `revision-${revision}`,
            },
            mediaType: "application/zip",
            size: 3,
          }),
          {
            status: revision === 1 ? 201 : 200,
            headers: {
              "content-type": "application/json",
              ETag: `"revision-${revision}"`,
            },
          },
        );
      }),
    );
    api.csrf.replace("a".repeat(43));
    const payload = new Blob(["zip"]);

    await writeProjectArtifact(api, {
      projectId: "project_example",
      namespace: "sources",
      name: "service",
      mediaType: "application/zip",
      payload,
    });
    await writeProjectArtifact(api, {
      projectId: "project_example",
      namespace: "sources",
      name: "service",
      mediaType: "application/zip",
      payload,
      expectedRevision: "revision-1",
    });

    expect(new URL(requests[0]!.url).pathname).toBe(
      "/v1/projects/project_example/artifacts/sources/service",
    );
    expect(requests[0]?.headers.get("If-None-Match")).toBe("*");
    expect(requests[1]?.headers.get("If-Match")).toBe('"revision-1"');
    const uploaded = await requests[0]!.text();
    expect(uploaded).toBe("zip");
    expect(uploaded.includes("owner_id")).toBe(false);
  });

  it("downloads one exact Project revision from Go Server", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response("zip", {
          headers: {
            "content-type": "application/zip",
            "content-length": "3",
          },
        });
      }),
    );
    await expect(
      downloadProjectArtifact(api, "project_example", metadata),
    ).resolves.toMatchObject({ mediaType: "application/zip" });
    const url = new URL(captured!.url);
    expect(url.pathname).toBe(
      "/v1/projects/project_example/artifacts/sources/service",
    );
    expect(url.searchParams.get("revision")).toBe("revision-1");
  });
});
