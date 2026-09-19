import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "./client";
import {
  archiveQueryKey,
  getArtifactArchive,
  getArtifactArchiveFile,
  type ArtifactArchiveScope,
} from "./artifact-archive";

const ref = { namespace: "skills", name: "example", revision: "r:1" };
const runtimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
const signal = () => new AbortController().signal;
const entry = { path: "SKILL.md", kind: "file", size: 4, previewable: true };

function response(value: unknown, etag = '"r:1"') {
  return new Response(JSON.stringify(value), {
    headers: {
      "Content-Type": "application/json",
      ETag: etag,
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

describe("Archive API", () => {
  it("explains index and individual file limits separately", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              code: "archive_preview_limit",
              message: "generic limit",
              retryable: false,
            }),
            {
              status: 413,
              headers: {
                "Content-Type": "application/json",
                "X-Contractor-API-Version": "contractor.public.v1",
              },
            },
          ),
      ),
    );
    await expect(
      getArtifactArchive(api, { kind: "user" }, ref, signal()),
    ).rejects.toMatchObject({
      status: 413,
      code: "archive_preview_limit",
      message: expect.stringContaining("browsing limits"),
    });
    await expect(
      getArtifactArchiveFile(api, { kind: "user" }, ref, "SKILL.md", signal()),
    ).rejects.toMatchObject({
      status: 413,
      code: "archive_preview_limit",
      message: expect.stringContaining("256 KiB"),
    });
  });
  it.each<{ scope: ArtifactArchiveScope; prefix: string }>([
    { scope: { kind: "user" }, prefix: "/v1" },
    {
      scope: { kind: "project", id: "project-one" },
      prefix: "/v1/projects/project-one",
    },
    { scope: { kind: "run", id: "run-one" }, prefix: "/v1/runs/run-one" },
  ])(
    "pins the revision and safely encodes file paths in $scope.kind scope",
    async ({ scope, prefix }) => {
      const requests: Request[] = [];
      const path = "references/file #1?&.md";
      const api = new PublicAPI(
        runtimeConfig,
        vi.fn(async (input) => {
          requests.push(input as Request);
          return response({ artifact: ref, path, size: 4, text: "text" });
        }),
      );
      await expect(
        getArtifactArchiveFile(api, scope, ref, path, signal()),
      ).resolves.toMatchObject({ path, text: "text" });
      const url = new URL(requests[0]!.url);
      expect(url.pathname).toBe(
        `${prefix}/artifacts/skills/example/archive/file`,
      );
      expect(url.searchParams.get("revision")).toBe("r:1");
      expect(url.searchParams.get("path")).toBe(path);
      expect([...url.searchParams.keys()]).toEqual(["revision", "path"]);
      expect(requests[0]!.credentials).toBe("include");
    },
  );

  it("keeps query identities separate across scopes and revisions", () => {
    const keys = [
      archiveQueryKey({ kind: "user" }, ref),
      archiveQueryKey({ kind: "project", id: "one" }, ref),
      archiveQueryKey({ kind: "project", id: "two" }, ref),
      archiveQueryKey({ kind: "run", id: "one" }, ref),
      archiveQueryKey({ kind: "user" }, { ...ref, revision: "r:2" }),
    ];
    expect(new Set(keys.map((key) => JSON.stringify(key))).size).toBe(
      keys.length,
    );
  });

  it.each([
    { artifact: { ...ref, revision: "other" }, entries: [entry] },
    { artifact: ref, entries: [entry, entry] },
    { artifact: ref, entries: [{ ...entry, path: "../secret" }] },
    { artifact: ref, entries: [{ ...entry, path: "missing/parent.txt" }] },
    { artifact: ref, entries: [{ ...entry, kind: "symlink" }] },
    { artifact: ref, entries: [{ ...entry, size: 262145 }] },
  ])("rejects malformed or mismatched directories: %j", async (body) => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () => response(body)),
    );
    await expect(
      getArtifactArchive(api, { kind: "user" }, ref, signal()),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
  });

  it("rejects a mismatched ETag and text byte count", async () => {
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({ artifact: ref, entries: [entry] }, '"other"'),
      ),
    );
    await expect(
      getArtifactArchive(api, { kind: "user" }, ref, signal()),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
    const textAPI = new PublicAPI(
      runtimeConfig,
      vi.fn(async () =>
        response({ artifact: ref, path: "SKILL.md", size: 1, text: "я" }),
      ),
    );
    await expect(
      getArtifactArchiveFile(
        textAPI,
        { kind: "user" },
        ref,
        "SKILL.md",
        signal(),
      ),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
  });

  it("rejects unsafe file selectors before making a request", async () => {
    const fetch = vi.fn();
    const api = new PublicAPI(runtimeConfig, fetch);
    for (const path of [
      "../secret",
      "a/../b",
      "/root",
      "C:\\file",
      "a\u202eb",
    ]) {
      await expect(
        getArtifactArchiveFile(api, { kind: "user" }, ref, path, signal()),
      ).rejects.toBeInstanceOf(TypeError);
    }
    expect(fetch).not.toHaveBeenCalled();
  });

  it("cancels an oversized streaming file response", async () => {
    const cancel = vi.fn();
    const api = new PublicAPI(
      runtimeConfig,
      vi.fn(
        async () =>
          new Response(
            new ReadableStream({
              pull(controller) {
                controller.enqueue(new Uint8Array(1024 * 1024));
              },
              cancel,
            }),
            {
              headers: {
                "Content-Type": "application/json",
                ETag: '"r:1"',
                "X-Contractor-API-Version": "contractor.public.v1",
              },
            },
          ),
      ),
    );
    await expect(
      getArtifactArchiveFile(api, { kind: "user" }, ref, "SKILL.md", signal()),
    ).rejects.toMatchObject({ code: "invalid_api_response" });
    expect(cancel).toHaveBeenCalledOnce();
  });
});
