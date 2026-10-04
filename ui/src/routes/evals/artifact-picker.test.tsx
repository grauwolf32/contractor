import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { createApplicationQueryClient } from "../../app/query-client";
import { sha256Hex } from "../../app/digest";
import { SessionProvider } from "../../auth/session";
import { EvalArtifactPicker } from "./artifact-picker";

const session = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-01T20:00:00Z",
  absoluteExpiresAt: "2026-09-02T12:00:00Z",
};

const input = {
  artifact: { namespace: "inputs", name: "spec", revision: "revision-1" },
  mediaType: "text/plain",
  size: 4,
  current: true,
  frozen: false,
  createdAt: "2026-09-01T10:00:00Z",
};

function reply(body: BodyInit, headers: Record<string, string>): Response {
  return new Response(body, {
    headers: { "X-Contractor-API-Version": "contractor.public.v1", ...headers },
  });
}

function json(value: unknown): Response {
  return reply(JSON.stringify(value), { "content-type": "application/json" });
}

function renderPicker(
  inventory: Record<"project" | "user", unknown[]>,
  onSelect: (value: unknown) => void,
) {
  const requests: Request[] = [];
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (value) => {
      const request = value instanceof Request ? value : new Request(value);
      requests.push(request);
      const url = new URL(request.url);
      if (url.pathname === "/v1/auth/session") return json(session);
      if (url.pathname === "/v1/projects/project_eval/artifacts")
        return json({ items: inventory.project, page: { hasMore: false } });
      if (url.pathname === "/v1/artifacts")
        return json({ items: inventory.user, page: { hasMore: false } });
      if (
        url.pathname === "/v1/artifacts/inputs/spec" ||
        url.pathname === "/v1/projects/project_eval/artifacts/inputs/spec"
      )
        return reply("spec", { "content-type": "text/plain" });
      throw new Error(`unexpected ${request.method} ${url}`);
    }),
  );
  render(
    <QueryClientProvider client={createApplicationQueryClient()}>
      <PublicAPIProvider api={api}>
        <SessionProvider api={api}>
          <EvalArtifactPicker projectId="project_eval" onSelect={onSelect} />
        </SessionProvider>
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return requests;
}

describe("Eval input picker", () => {
  it("pins My artifacts inputs through the User Artifact API without Skill packages", async () => {
    const onSelect = vi.fn();
    const requests = renderPicker({ project: [], user: [input] }, onSelect);
    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("Input source"), "user");
    await user.click(await screen.findByRole("button", { name: "Use input" }));

    await waitFor(() => expect(onSelect).toHaveBeenCalledTimes(1));
    expect(onSelect.mock.calls[0]?.[0]).toEqual({
      scope: "user",
      scopeId: "user_local",
      namespace: "inputs",
      name: "spec",
      revision: "revision-1",
      sha256: `sha256:${await sha256Hex(new TextEncoder().encode("spec"))}`,
      mediaType: "text/plain",
      sizeBytes: 4,
    });
    const list = requests.find(
      (request) => new URL(request.url).pathname === "/v1/artifacts",
    );
    expect(
      new URL(list?.url ?? "http://invalid").searchParams.get(
        "excludeNamespace",
      ),
    ).toBe("skills");
    const download = requests.at(-1);
    expect(download?.url).toBe(
      "http://127.0.0.1:8080/v1/artifacts/inputs/spec?revision=revision-1",
    );
    expect(download?.headers.get("Accept")).toBe("text/plain");
  });

  it("validates a workspace input identity before downloading it", async () => {
    const onSelect = vi.fn();
    const requests = renderPicker(
      {
        project: [
          {
            ...input,
            artifact: { ...input.artifact, revision: "bad revision" },
          },
        ],
        user: [],
      },
      onSelect,
    );
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Use input" }));

    expect(
      await screen.findByText("Project Artifact revision is invalid"),
    ).toBeVisible();
    expect(onSelect).not.toHaveBeenCalled();
    expect(
      requests.some((request) =>
        new URL(request.url).pathname.endsWith("/inputs/spec"),
      ),
    ).toBe(false);
  });
});
