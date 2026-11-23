import cases from "../../../api/testdata/git-import-modes.json";
import { conformsTo } from "./contract";

import { describe, expect, it, vi } from "vitest";

import { fakeServer } from "../test/fake-server";
import { json } from "../test/responses";
import { testSharedConfig } from "../test/shared-config";
import { testAPI } from "../test/render";
import { importGitArtifact } from "./git-artifacts";

describe("importGitArtifact", () => {
  const draft = {
    namespace: "sources",
    name: "service",
    repositoryUrl: "https://git.example.test/service.git",
  };

  it.each(["", "-r1", "r 1", '"r1"', "r".repeat(257)])(
    "refuses expected revision %j before sending a request",
    async (expectedRevision) => {
      const server = fakeServer({
        "POST /v1/artifacts/{namespace}/{name}/git-import": () => json({}, 201),
      });
      const fetch = vi.spyOn(server, "fetch");
      const { api } = testAPI(server);
      await expect(
        importGitArtifact(
          api,
          { ...draft, expectedRevision },
          new AbortController().signal,
        ),
      ).rejects.toMatchObject({
        code: "invalid_client_request",
        message: expect.stringContaining("If-Match"),
      });
      expect(server.requests).toHaveLength(0);
      expect(fetch).not.toHaveBeenCalled();
    },
  );
});

it.each(cases.cases)("validates shared Git import mode $name", (test) => {
  expect(conformsTo("GitImportRequest", test.request)).toBe(test.valid);
});

it.each(cases.analysisSummaries)(
  "validates shared analysis summary $name",
  (test) => {
    expect(conformsTo("GitAnalysis", test.value)).toBe(test.valid);
  },
);

it("uses the received Git URL byte ceiling in the API client", async () => {
  const url = "https://example.test/é";
  const maxBytes = new TextEncoder().encode(url).length;
  const result = {
    artifact: { namespace: "sources", name: "repo", revision: "r1" },
    mediaType: "application/zip",
    size: 1,
    gitSource: {
      repositoryUrl: "https://example.test:443/%C3%A9",
      requestedRef: "main",
      resolvedCommit: "a".repeat(40),
      importedAt: "2026-10-31T00:00:00Z",
    },
  };
  const server = fakeServer({
    "GET /v1/config": () =>
      json(testSharedConfig(17, { gitImportUrlMaxBytes: maxBytes })),
    "POST /v1/artifacts/{namespace}/{name}/git-import": () => json(result, 201),
  });
  const { api } = testAPI(server);
  const draft = { namespace: "sources", name: "repo", repositoryUrl: url };
  await expect(
    importGitArtifact(api, draft, new AbortController().signal),
  ).resolves.toEqual(result);
  await expect(
    importGitArtifact(
      api,
      { ...draft, repositoryUrl: url + "x" },
      new AbortController().signal,
    ),
  ).rejects.toThrow("UTF-8 bytes");
  expect(
    server.sent("POST", "/v1/artifacts/sources/repo/git-import"),
  ).toHaveLength(1);
});
