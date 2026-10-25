import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";

import { renderApplication } from "../../test/application";
import { project } from "../../test/builders";
import { createEvalFixture } from "../../test/evals-fixture";
import { json, page } from "../../test/responses";
import { evalServer, stubEvalBrowser } from "./test-support";

beforeEach(stubEvalBrowser);

describe("Current evaluation workspaces", () => {
  it("recovers the same workspace creation across collection and setup without creating an experiment", async () => {
    const fixture = createEvalFixture();
    const server = evalServer(fixture);
    const workspace = project("evaluation-new", {
      kind: "evaluation",
      name: "New comparison workspace",
      description: "Shared cases",
    });
    const submissions: { key: string | null; body: unknown }[] = [];
    let created = false;
    server.route("GET /v1/projects", () =>
      json(page(created ? [workspace] : [])),
    );
    server.route("POST /v1/projects", async ({ request }) => {
      submissions.push({
        key: request.headers.get("Idempotency-Key"),
        body: await request.json(),
      });
      created = true;
      if (submissions.length === 1) throw new TypeError("Response lost");
      return json(workspace, 201, { ETag: '"1"' });
    });
    const user = userEvent.setup();
    let view = renderApplication("/evals/workspaces", { server });
    await user.click(
      (await screen.findAllByRole("button", { name: "New workspace" }))[0]!,
    );
    let dialog = screen.getByRole("dialog", {
      name: "New evaluation workspace",
    });
    await user.type(within(dialog).getByLabelText("Name"), workspace.name);
    await user.type(
      within(dialog).getByLabelText("Description"),
      workspace.description,
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Create workspace" }),
    );
    await screen.findByText(/Public API is unavailable/);
    view.unmount();

    view = renderApplication("/evals/new", { server });
    await user.type(
      await screen.findByLabelText("Experiment name"),
      "Keep this draft",
    );
    await user.click(screen.getByRole("button", { name: "New workspace" }));
    dialog = screen.getByRole("dialog", { name: "New evaluation workspace" });
    expect(within(dialog).getByLabelText("Name")).toHaveAttribute(
      "maxlength",
      "160",
    );
    await user.type(within(dialog).getByLabelText("Name"), workspace.name);
    await user.type(
      within(dialog).getByLabelText("Description"),
      workspace.description,
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Create workspace" }),
    );
    await waitFor(() =>
      expect(screen.getByLabelText("Evaluation workspace")).toHaveValue(
        workspace.projectId,
      ),
    );
    expect(screen.getByLabelText("Experiment name")).toHaveValue(
      "Keep this draft",
    );
    expect(
      screen.getByRole("link", { name: "Manage workspace" }),
    ).toHaveAttribute("href", `/evals/workspaces/${workspace.projectId}`);
    expect(submissions).toHaveLength(2);
    expect(submissions[1]).toEqual(submissions[0]);
    expect(submissions[0]!.key).toMatch(/^eval-ui-/);
    expect(server.sent("POST", "/eval-experiments")).toHaveLength(0);
    expect(view.router.state.location.pathname).toBe("/evals/new");
  });

  it.each([
    "/evals/experiments/experiment-1/overview",
    "/evals/datasets?project=evaluation-1",
  ])("links %s to its current workspace", async (path) => {
    const fixture = createEvalFixture();
    renderApplication(path, { server: evalServer(fixture) });
    const links = await screen.findAllByRole("link", {
      name: "Manage workspace",
    });
    expect(links[0]).toHaveAttribute("href", "/evals/workspaces/evaluation-1");
  });
});
