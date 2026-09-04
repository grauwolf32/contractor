import { describe, expect, it, vi } from "vitest";

import type { RuntimeConfig } from "../config/runtime-config";
import {
  canPreviewArtifact,
  downloadArtifact,
  listArtifacts,
  MAXIMUM_PREVIEW_BYTES,
  previewArtifact,
  suggestedArtifactFilename,
  writeArtifact,
  type ArtifactMetadata,
} from "./artifacts";
import { PublicAPI } from "./client";

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

const metadata: ArtifactMetadata = {
  artifact: {
    namespace: "projects",
    name: "architecture",
    revision: "revision-2",
  },
  mediaType: "text/vnd.likec4",
  size: 17,
  current: true,
  frozen: false,
  createdAt: "2026-08-31T12:00:00Z",
};

function response(body: BodyInit | null, options: ResponseInit = {}): Response {
  const headers = new Headers(options.headers);
  headers.set("X-Contractor-API-Version", "contractor.public.v1");
  return new Response(body, { ...options, headers });
}

describe("Artifact API", () => {
  it("uses generated list parameters and maps the authoritative page", async () => {
    let captured: Request | undefined;
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async (input) => {
        captured = input instanceof Request ? input : new Request(input);
        return response(
          JSON.stringify({
            items: [metadata],
            page: { hasMore: true, nextCursor: "cursor-next" },
          }),
          { headers: { "content-type": "application/json" } },
        );
      }),
    );
    await expect(
      listArtifacts(api, { namespace: "projects", cursor: "cursor-1" }),
    ).resolves.toMatchObject({ items: [metadata] });
    const url = new URL(captured?.url ?? "http://invalid");
    expect(url.pathname).toBe("/v1/artifacts");
    expect(url.searchParams.get("namespace")).toBe("projects");
    expect(url.searchParams.get("cursor")).toBe("cursor-1");
    expect(url.searchParams.get("limit")).toBe("50");
  });

  it("creates and updates with mutually exclusive exact CAS headers", async () => {
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
              namespace: "projects",
              name: "source",
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
    await writeArtifact(api, {
      namespace: "projects",
      name: "source",
      mediaType: "application/zip",
      payload,
    });
    await writeArtifact(api, {
      namespace: "projects",
      name: "source",
      mediaType: "application/zip",
      payload,
      expectedRevision: "revision-1",
    });

    expect(requests[0]?.headers.get("If-None-Match")).toBe("*");
    expect(requests[0]?.headers.has("If-Match")).toBe(false);
    expect(requests[1]?.headers.get("If-Match")).toBe('"revision-1"');
    expect(requests[1]?.headers.has("If-None-Match")).toBe(false);
    expect(requests[1]?.headers.get("X-CSRF-Token")).toBe("a".repeat(43));
    await expect(requests[0]?.text()).resolves.toBe("zip");
  });

  it("surfaces conflict once and never retries an unsafe write", async () => {
    const fetchImplementation = vi.fn(async () =>
      response(
        JSON.stringify({
          code: "conflict",
          message: "resource state changed; retry with the current revision",
          retryable: true,
          requestId: "request-conflict",
        }),
        { status: 409, headers: { "content-type": "application/json" } },
      ),
    );
    const api = new PublicAPI(runtimeConfig, fetchImplementation);
    api.csrf.replace("a".repeat(43));
    await expect(
      writeArtifact(api, {
        namespace: "projects",
        name: "source",
        mediaType: "text/plain",
        payload: new Blob(["next"]),
        expectedRevision: "revision-stale",
      }),
    ).rejects.toMatchObject({
      code: "conflict",
      requestId: "request-conflict",
    });
    expect(fetchImplementation).toHaveBeenCalledTimes(1);
  });

  it("previews only exact bounded allowlisted UTF-8 text", async () => {
    const content = "specification {\n}";
    const exactMetadata = { ...metadata, size: content.length };
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response(content, {
          headers: {
            "content-type": "text/vnd.likec4",
            "content-length": String(content.length),
          },
        }),
      ),
    );
    await expect(previewArtifact(api, exactMetadata)).resolves.toBe(content);
    expect(canPreviewArtifact(exactMetadata)).toBe(true);
    expect(
      canPreviewArtifact({ ...exactMetadata, mediaType: "text/x-diff" }),
    ).toBe(true);
    expect(
      canPreviewArtifact({
        ...exactMetadata,
        size: MAXIMUM_PREVIEW_BYTES + 1,
      }),
    ).toBe(false);
    expect(
      canPreviewArtifact({ ...exactMetadata, mediaType: "text/html" }),
    ).toBe(false);
  });

  it("rejects download metadata drift and creates a sanitized filename", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response("wrong", {
          headers: {
            "content-type": "text/plain",
            "content-length": "5",
          },
        }),
      ),
    );
    await expect(downloadArtifact(api, metadata)).rejects.toMatchObject({
      code: "invalid_api_response",
    });
    expect(
      suggestedArtifactFilename(metadata.artifact, metadata.mediaType),
    ).toBe("projects-architecture-revision-2.c4");
    expect(suggestedArtifactFilename(metadata.artifact, "text/x-diff")).toBe(
      "projects-architecture-revision-2.diff",
    );
  });
});
