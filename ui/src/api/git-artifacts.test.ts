import { describe, expect, it, vi } from "vitest";

import { fakeServer, json } from "../test/fake-server";
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
