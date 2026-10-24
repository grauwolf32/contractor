import { act, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { sweepKeys } from "../../api/scan-sweeps";
import { project } from "../../test/builders";
import { fakeServer } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { failure, json, noContent, page } from "../../test/responses";
import {
  scanSweep,
  sweepMember,
  webScanPack,
  webScanWorkflow,
} from "../../test/web-scans";
import { NewSweepRoute } from "./sweep-new";
import { RuleSweepsRoute } from "./sweeps";

describe("rule sweep widgets", () => {
  it("captures selections across Project pages with an exact pack and replays one request after a lost acknowledgement", async () => {
    const server = fakeServer({
      "GET /v1/workflows/{name}/versions/{version}": () =>
        json(webScanWorkflow),
      "GET /v1/projects": ({ url }) =>
        json(
          url.searchParams.has("cursor")
            ? page([project("later", { name: "Later Project" })])
            : page([project("first", { name: "First Project" })], "next"),
        ),
      "GET /v1/artifacts": () => json(page([webScanPack])),
    });
    let attempts = 0;
    server.route("POST /v1/scan-sweeps", () =>
      ++attempts === 1
        ? failure(503, "storage_transaction_conflict", "Retry sweep", true)
        : json(scanSweep(), 201),
    );
    const view = renderWithAPI(<NewSweepRoute />, { server });
    await view.user.click(
      await screen.findByRole("checkbox", { name: /First Project/ }),
    );
    await view.user.click(
      screen.getByRole("button", { name: "Load more Projects" }),
    );
    await view.user.click(
      await screen.findByRole("checkbox", { name: /Later Project/ }),
    );
    await view.user.selectOptions(
      screen.getByLabelText("Rule source"),
      "custom",
    );
    await screen.findByRole("option", { name: "rules/web-check · r1" });
    await view.user.selectOptions(
      screen.getByLabelText(/Rulepack from my library/),
      "rules/web-check@r1",
    );
    expect(screen.queryByLabelText(/Target URL/)).not.toBeInTheDocument();
    await view.user.type(screen.getByLabelText("Rule IDs"), "fixture");
    await view.user.click(screen.getByRole("button", { name: "Start sweep" }));
    await screen.findByText("Retry sweep");
    await view.user.click(screen.getByRole("button", { name: "Start sweep" }));
    await waitFor(() =>
      expect(view.location()).toBe("/checks/sweeps/sweep-one"),
    );
    const requests = server.sent("POST", "/scan-sweeps");
    expect(requests).toHaveLength(2);
    expect(requests[0]?.headers.get("Idempotency-Key")).toBe(
      requests[1]?.headers.get("Idempotency-Key"),
    );
    expect(await requests[0]?.json()).toEqual({
      workflow: "nuclei-web@1",
      projectIds: ["first", "later"],
      parameters: { template_ids: "fixture" },
      rulepack: webScanPack.artifact,
    });
  });
  it("sends an explicit all-owner scope without guessing a partial Project list", async () => {
    const server = fakeServer({
      "GET /v1/workflows/{name}/versions/{version}": () =>
        json(webScanWorkflow),
      "GET /v1/projects": () => json(page([])),
      "POST /v1/scan-sweeps": () => json(scanSweep(), 201),
    });
    const view = renderWithAPI(<NewSweepRoute />, { server });
    await view.user.selectOptions(
      await screen.findByLabelText("Project scope"),
      "all",
    );
    await view.user.click(screen.getByRole("button", { name: "Start sweep" }));
    await waitFor(() =>
      expect(server.sent("POST", "/scan-sweeps")).toHaveLength(1),
    );
    expect(await server.sent("POST", "/scan-sweeps")[0]?.json()).toEqual({
      workflow: "nuclei-web@1",
      allProjects: true,
      parameters: {},
    });
  });
  it("shows actual cancellation intent and refreshes members once the parent settles", async () => {
    let current = scanSweep();
    let settled = false;
    const server = fakeServer({
      "GET /v1/scan-sweeps": () => json(page([current])),
      "GET /v1/scan-sweeps/{sweepId}": () => json(current),
      "GET /v1/scan-sweeps/{sweepId}/members": () =>
        json(page([sweepMember(0, settled)])),
      "POST /v1/scan-sweeps/{sweepId}/cancel": () => {
        current = scanSweep("cancelling");
        return json(current);
      },
    });
    const view = renderWithAPI(null, {
      server,
      path: "/checks/sweeps/sweep-one",
      routes: [
        { path: "/checks/sweeps/:sweepId", element: <RuleSweepsRoute /> },
      ],
    });
    await view.user.click(
      await screen.findByRole("button", { name: "Cancel sweep" }),
    );
    await screen.findByText("Cancelling", { exact: true });
    expect(
      screen.queryByText("Cancelled", {
        exact: true,
        selector: ".ui-status-chip",
      }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Delete sweep" }),
    ).not.toBeInTheDocument();
    current = scanSweep("completed");
    settled = true;
    await act(async () => {
      await view.queryClient.invalidateQueries({
        queryKey: sweepKeys.detail("sweep-one"),
      });
    });
    await screen.findByText("Completed", { exact: true });
    expect(
      await screen.findByRole("link", { name: "Open scan and results" }),
    ).toHaveAttribute("href", "/runs/run-0");
    expect(screen.getByText(/Run Failed/)).toBeInTheDocument();
  });
  it("pages only the visible members, and confirms purge without deleting child Runs", async () => {
    const server = fakeServer({
      "GET /v1/scan-sweeps": () => json(page([scanSweep("completed")])),
      "GET /v1/scan-sweeps/{sweepId}": () => json(scanSweep("completed")),
      "GET /v1/scan-sweeps/{sweepId}/members": ({ url }) =>
        json(
          url.searchParams.has("cursor")
            ? page([sweepMember(50, true)])
            : page(
                Array.from({ length: 50 }, (_, i) => sweepMember(i)),
                "members-next",
              ),
        ),
      "DELETE /v1/scan-sweeps/{sweepId}": () => noContent(),
    });
    const view = renderWithAPI(null, {
      server,
      path: "/checks/sweeps/sweep-one",
      routes: [
        { path: "/checks/sweeps/:sweepId", element: <RuleSweepsRoute /> },
      ],
    });
    const pagination = await screen.findByRole("navigation", {
      name: "Member pages",
    });
    await view.user.click(
      within(pagination).getByRole("button", { name: "Next" }),
    );
    expect(
      await screen.findByRole("link", { name: "Open scan and results" }),
    ).toHaveAttribute("href", "/runs/run-50");
    expect(
      screen.queryByRole("link", { name: /^p0$/ }),
    ).not.toBeInTheDocument();
    await view.user.click(screen.getByRole("button", { name: "Delete sweep" }));
    expect(server.sent("DELETE", "/sweep-one")).toHaveLength(0);
    const dialog = screen.getByRole("alertdialog");
    await view.user.click(
      within(dialog).getByRole("button", { name: "Delete sweep" }),
    );
    await waitFor(() => expect(view.location()).toBe("/checks/sweeps"));
    expect(
      server.requests
        .filter((request) => request.method === "DELETE")
        .map((request) => new URL(request.url).pathname),
    ).toEqual(["/v1/scan-sweeps/sweep-one"]);
  });
});
