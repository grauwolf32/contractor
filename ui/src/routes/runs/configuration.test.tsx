import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";

function setup(path: string, authorized = true) {
  const requests: string[] = [];
  const runtimeResource = {
    ref: {
      name: "debug",
      version: "1",
      digest: `sha256:${"1".repeat(64)}`,
    },
    document: {
      apiVersion: "contractor/v1alpha1",
      kind: "RuntimeConfig",
      metadata: { name: "debug", version: "1" },
      spec: {
        worker: {
          caido: {
            adapter: "caido-graphql@1",
            endpoint: "https://caido.example/graphql",
            credential: "caido-local",
            requestTimeoutSeconds: 30,
          },
        },
      },
    },
    builtIn: false,
    createdBy: "user_local",
    createdAt: "2026-09-07T00:00:00Z",
  };
  const api = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const pathname = new URL(request.url).pathname;
      requests.push(pathname);
      let body: unknown = { items: [], page: { hasMore: false } };
      if (pathname === "/v1/auth/session")
        body = {
          principal: {
            userId: "user_local",
            username: "owner",
            capabilities: authorized ? ["user", "operations"] : ["user"],
          },
          csrfToken: "a".repeat(43),
          idleExpiresAt: "2026-09-07T20:00:00Z",
          absoluteExpiresAt: "2026-09-08T12:00:00Z",
        };
      else if (pathname === "/v1/operations/snapshot")
        body = {
          cursor: { generation: "operations-generation-1", revision: "1" },
          runtimeAgents: [],
          allocations: [],
        };
      else if (pathname === "/v1/operations/runtime-configs")
        body = { items: [runtimeResource], page: { hasMore: false } };
      else if (pathname === "/v1/operations/runtime-configs/debug/versions/1")
        body = runtimeResource;
      return new Response(JSON.stringify(body), {
        headers: {
          "content-type": "application/json",
          "X-Contractor-API-Version": "contractor.public.v1",
        },
      });
    }),
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  render(<Application api={api} publicAPI={api} router={router} />);
  return { router, requests };
}

describe("Run configuration navigation", () => {
  it("opens creation forms from icons and clears a closed credential draft", async () => {
    setup("/runs/configuration");
    const user = userEvent.setup();
    const addCredential = await screen.findByRole("button", {
      name: "Add Runtime credential",
    });
    expect(screen.queryByLabelText("Runtime credential ID")).toBeNull();
    expect(screen.queryByLabelText("RuntimeConfig name")).toBeNull();
    expect(
      screen.queryByRole("heading", { name: "Runtime label bindings" }),
    ).toBeNull();
    await user.click(addCredential);
    let dialog = screen.getByRole("dialog", {
      name: "Create Runtime credential",
    });
    expect(
      within(dialog).getByLabelText("Runtime credential ID"),
    ).toHaveFocus();
    await user.type(
      within(dialog).getByLabelText(/Header value/),
      "temporary-secret",
    );
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    await user.click(addCredential);
    dialog = screen.getByRole("dialog", { name: "Create Runtime credential" });
    expect(within(dialog).getByLabelText(/Header value/)).toHaveValue("");
    await user.click(
      within(dialog).getByRole("button", {
        name: "Close Runtime credential form",
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Publish RuntimeConfig" }),
    );
    expect(
      screen.getByRole("dialog", { name: "Publish RuntimeConfig" }),
    ).toBeVisible();
    expect(screen.getByLabelText("RuntimeConfig name")).toHaveFocus();
    await user.keyboard("{Escape}");
    await user.click(
      await screen.findByRole("button", {
        name: "Manage bindings for debug@1",
      }),
    );
    expect(
      screen.getByRole("dialog", { name: "Runtime label bindings" }),
    ).toBeVisible();
    expect(screen.getByLabelText("RuntimeConfig")).toHaveDisplayValue(
      "debug@1",
    );
  });

  it("opens Runtime configuration under Runs and supports refresh", async () => {
    const { router, requests } = setup("/runs/configuration");
    await screen.findByRole("heading", { name: "RuntimeConfig versions" });
    expect(await screen.findByRole("link", { name: "debug@1" })).toBeVisible();
    expect(router.state.location.pathname).toBe("/runs/configuration");
    const tabs = screen.getByRole("navigation", { name: "Run views" });
    expect(tabs).toHaveClass("operations-navigation");
    expect(
      within(tabs)
        .getAllByRole("link")
        .map((link) => link.textContent?.trim()),
    ).toEqual(["Queue", "Completed", "Configuration"]);
    expect(
      within(tabs).getByRole("link", { name: /Configuration/ }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(tabs).getByRole("link", { name: /Queue/ }),
    ).not.toHaveAttribute("aria-current");
    expect(
      screen.queryByRole("navigation", { name: "Operations sections" }),
    ).toBeNull();
    const reads = requests.filter(
      (p) => p === "/v1/operations/runtime-configs",
    ).length;
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(
        requests.filter((p) => p === "/v1/operations/runtime-configs").length,
      ).toBeGreaterThan(reads),
    );
  });

  it("opens exact versions with query and fragment", async () => {
    const { router } = setup(
      "/runs/configuration/debug/1?from=bookmark#worker",
    );
    await screen.findByRole("heading", { name: "debug@1" });
    expect(screen.getByRole("heading", { name: "Worker Caido" })).toBeVisible();
    expect(screen.getByText("https://caido.example/graphql")).toBeVisible();
    expect(router.state.location).toMatchObject({
      pathname: "/runs/configuration/debug/1",
      search: "?from=bookmark",
      hash: "#worker",
    });
    expect(
      screen.getByRole("link", { name: /← Runtime configuration/ }),
    ).toHaveAttribute("href", "/runs/configuration");
  });

  it.each(["/runs/configuration", "/runs/configuration/debug/1"])(
    "keeps operator authorization on %s",
    async (path) => {
      const { requests } = setup(path, false);
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Operations capability is required",
      );
      expect(screen.queryByRole("link", { name: /Configuration/ })).toBeNull();
      expect(requests).toEqual(["/v1/auth/session"]);
    },
  );
});
