import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { ArtifactMetadata } from "../../api/artifacts";
import { sha256Hex } from "../../app/digest";
import { conforming } from "../../test/builders";
import { fakeServer, invalidJSON } from "../../test/fake-server";
import { bytes, json, page } from "../../test/responses";
import { renderWithAPI, testSession } from "../../test/render";
import { EvalArtifactPicker } from "./artifact-picker";

const input = conforming("ArtifactMetadata", {
  artifact: { namespace: "inputs", name: "spec", revision: "revision-1" },
  mediaType: "text/plain",
  size: 4,
  current: true,
  frozen: false,
  createdAt: "2026-09-01T10:00:00Z",
});

/** The picker over Project and User inventories; `broken` breaks the contract on purpose. */
function renderPicker(
  inventory: Record<"project" | "user", ArtifactMetadata[]>,
  onSelect: (value: unknown) => void,
  broken = false,
) {
  const list = (items: ArtifactMetadata[]) =>
    (broken ? invalidJSON : json)(page(items));
  const download = () => bytes("spec", "text/plain");
  const server = fakeServer({
    "GET /v1/projects/{projectId}/artifacts": () => list(inventory.project),
    "GET /v1/artifacts": () => list(inventory.user),
    "GET /v1/artifacts/{namespace}/{name}": download,
    "GET /v1/projects/{projectId}/artifacts/{namespace}/{name}": download,
  });
  renderWithAPI(
    <EvalArtifactPicker projectId="project_eval" onSelect={onSelect} />,
    { server, session: testSession(["user"]) },
  );
  return server.requests;
}

describe("Eval input picker", () => {
  it("says when a source has no files, without an empty list", async () => {
    renderPicker({ project: [], user: [] }, vi.fn());
    expect(await screen.findByText("No files here yet.")).toBeVisible();
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
  });

  it("pins inputs from My files through the User Artifact API without Skill packages", async () => {
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

  it("offers no input from a list whose identity breaks the contract", async () => {
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
      true,
    );

    expect(
      await screen.findByText(
        /The Server sent an answer this page cannot read/,
      ),
    ).toBeVisible();
    expect(screen.queryByRole("button", { name: "Use input" })).toBeNull();
    expect(onSelect).not.toHaveBeenCalled();
    expect(
      requests.some((request) =>
        new URL(request.url).pathname.endsWith("/inputs/spec"),
      ),
    ).toBe(false);
  });
});
