import { screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { project } from "../../test/builders";
import { fakeServer } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { failure, json, page } from "../../test/responses";
import {
  webScanCreated,
  webScanPack,
  webScanWorkflow,
} from "../../test/web-scans";
import { WebScanRoute } from "./new";

function fixture() {
  return fakeServer({
    "GET /v1/workflows/{name}/versions/{version}": () => json(webScanWorkflow),
    "GET /v1/projects": ({ url }) =>
      json(
        url.searchParams.has("cursor")
          ? page([
              project("project-later", {
                name: "Later Project",
                httpTarget: { url: "https://fixture.example/" },
              }),
            ])
          : page(
              Array.from({ length: 50 }, (_, i) => project(`p${i}`)),
              "projects-next",
            ),
      ),
    "GET /v1/artifacts": ({ url }) =>
      json(
        url.searchParams.has("cursor")
          ? page([webScanPack])
          : page(
              Array.from({ length: 50 }, (_, i) => ({
                ...webScanPack,
                artifact: { ...webScanPack.artifact, name: `other-${i}` },
                mediaType: "text/plain",
              })),
              "packs-next",
            ),
      ),
  });
}
describe("basic web scan launcher", () => {
  it("selects later Project and owner library pages, keeping exact inputs and strings on lost-response retry", async () => {
    const server = fixture();
    let attempts = 0;
    server.route("POST /v1/projects/{projectId}/runs", () =>
      ++attempts === 1
        ? failure(503, "storage_transaction_conflict", "Retry later", true)
        : json(webScanCreated("project-later"), 202),
    );
    const view = renderWithAPI(<WebScanRoute />, {
      path: "/checks/web/new",
      server,
    });
    await screen.findByRole("button", { name: "Load more Projects" });
    await view.user.click(
      screen.getByRole("button", { name: "Load more Projects" }),
    );
    await screen.findByRole("option", { name: "Later Project" });
    await view.user.selectOptions(
      screen.getByLabelText("Save results in"),
      "project-later",
    );
    expect(screen.getByLabelText(/Target URL/)).toHaveValue(
      "https://fixture.example/",
    );
    await view.user.selectOptions(
      screen.getByLabelText("Rule source"),
      "custom",
    );
    await view.user.click(
      await screen.findByRole("button", { name: "Load more library files" }),
    );
    await screen.findByRole("option", { name: "rules/web-check · r1" });
    await view.user.selectOptions(
      screen.getByLabelText(/Rulepack from my library/),
      "rules/web-check@r1",
    );
    await view.user.type(screen.getByLabelText("Rule IDs"), "CVE-fixture");
    await view.user.type(screen.getByLabelText("Requests per second"), "010");
    await view.user.click(screen.getByRole("button", { name: "Start scan" }));
    await screen.findByText("Retry later");
    expect(screen.getByLabelText("Requests per second")).toHaveValue("010");
    await view.user.click(screen.getByRole("button", { name: "Start scan" }));
    await waitFor(() => expect(view.location()).toBe("/runs/run-web"));
    const sent = server.sent("POST", "/runs");
    expect(sent).toHaveLength(2);
    expect(sent[0]?.headers.get("Idempotency-Key")).toBe(
      sent[1]?.headers.get("Idempotency-Key"),
    );
    expect(await sent[0]?.json()).toEqual({
      workflow: "nuclei-web@1",
      parameters: {
        target: "https://fixture.example/",
        template_ids: "CVE-fixture",
        rate_limit: "010",
      },
      artifacts: { rulepack: webScanPack.artifact },
    });
    expect(
      server.requests.some(
        (request) =>
          new URL(request.url).pathname.includes("/artifacts/") &&
          new URL(request.url).pathname.includes("/projects/"),
      ),
    ).toBe(false);
  });
  it("launches standalone with omitted defaults and a fresh key after a reviewed edit", async () => {
    const server = fixture();
    server.route("POST /v1/runs", () =>
      failure(400, "invalid_request", "Invalid rate"),
    );
    const view = renderWithAPI(<WebScanRoute />, { server });
    await view.user.type(
      await screen.findByLabelText(/Target URL/),
      "https://example.com",
    );
    await view.user.click(screen.getByRole("button", { name: "Start scan" }));
    await screen.findByText("Invalid rate");
    await view.user.type(screen.getByLabelText("Rule IDs"), "changed");
    await view.user.click(screen.getByRole("button", { name: "Start scan" }));
    await waitFor(() =>
      expect(server.sent("POST", "/v1/runs")).toHaveLength(2),
    );
    const sent = server.sent("POST", "/v1/runs");
    expect(sent[0]?.headers.get("Idempotency-Key")).not.toBe(
      sent[1]?.headers.get("Idempotency-Key"),
    );
    expect(await sent[0]?.json()).toEqual({
      workflow: "nuclei-web@1",
      parameters: { target: "https://example.com" },
    });
  });
  it("keeps a failed continuation visible and retries it without creating a scan", async () => {
    const server = fixture();
    server.route("GET /v1/artifacts", ({ url }) =>
      url.searchParams.has("cursor")
        ? failure(503, "storage_unavailable", "Library unavailable")
        : json(page([], "next")),
    );
    const view = renderWithAPI(<WebScanRoute />, { server });
    await view.user.selectOptions(
      await screen.findByLabelText("Rule source"),
      "custom",
    );
    await view.user.click(
      await screen.findByRole("button", { name: "Load more library files" }),
    );
    await screen.findByText("Library unavailable");
    expect(screen.getByRole("button", { name: "Start scan" })).toBeDisabled();
    expect(server.sent("POST", "/v1/runs")).toHaveLength(0);
  });
});
