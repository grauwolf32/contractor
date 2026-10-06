import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../../api/client";
import { Application } from "../../../app/application";
import { applicationRoutes } from "../../../app/router";

function setup(
  path: string,
  authorized = true,
  withRemovalFixtures = false,
  workerSettings?: Record<string, unknown>,
  deleteFailure?: { path: string; body: unknown },
) {
  const requests: string[] = [];
  const deletes: Request[] = [];
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
        worker: workerSettings ?? {
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
  let credentials = withRemovalFixtures
    ? [
        {
          credentialId: "caido-local",
          kind: "caido-bearer@1",
          createdBy: "user_local",
          createdAt: "2026-09-07T00:00:00Z",
        },
      ]
    : [];
  let bindings = withRemovalFixtures
    ? [
        {
          label: "debug",
          config: runtimeResource.ref,
          revision: "1",
          createdBy: "user_local",
          createdAt: "2026-09-07T00:00:00Z",
          updatedBy: "user_local",
          updatedAt: "2026-09-07T00:00:00Z",
        },
      ]
    : [];
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
      if (request.method === "DELETE") {
        deletes.push(request.clone());
        if (deleteFailure?.path === pathname) {
          return new Response(JSON.stringify(deleteFailure.body), {
            status: 409,
            headers: {
              "content-type": "application/json",
              "X-Contractor-API-Version": "contractor.public.v1",
            },
          });
        }
        if (pathname === "/v1/operations/runtime-credentials/caido-local") {
          credentials = [];
        } else if (pathname === "/v1/operations/runtime-labels/debug") {
          bindings = [];
        } else {
          throw new Error(`unexpected DELETE ${pathname}`);
        }
        return new Response(null, {
          status: 204,
          headers: { "X-Contractor-API-Version": "contractor.public.v1" },
        });
      }
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
      else if (pathname === "/v1/operations/runtime-credentials")
        body = { items: credentials, page: { hasMore: false } };
      else if (pathname === "/v1/operations/runtime-labels")
        body = { items: bindings, page: { hasMore: false } };
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
  return { router, requests, deletes };
}

describe("Runtime configuration hub navigation", () => {
  it("shows nested RuntimeConfig export settings and explicit clears with labels", async () => {
    setup("/operations/configuration/debug/1", true, false, {
      llmGateway: {
        gateway: {
          gatewayId: "local-litellm",
          version: "1",
          digest: `sha256:${"2".repeat(64)}`,
        },
        credential: null,
      },
      telemetry: {
        adapter: "otlp-http@1",
        endpoint: "https://otel.example/v1/traces",
        export: {
          batchSizeBytes: 8388608,
          maxAttempts: 2,
          maxPendingBytes: 67108864,
          maxPendingSpans: 2048,
          retry: {
            initialBackoffMilliseconds: 100,
            maxBackoffMilliseconds: 1000,
          },
        },
      },
    });
    const telemetry = (
      await screen.findByRole("heading", { name: "Worker telemetry" })
    ).closest("article");
    expect(telemetry).not.toBeNull();
    expect(within(telemetry!).getByText("Batch size")).toBeVisible();
    expect(within(telemetry!).getByText("8.0 MiB")).toBeVisible();
    expect(within(telemetry!).getByText("Maximum pending bytes")).toBeVisible();
    expect(within(telemetry!).getByText("64.0 MiB")).toBeVisible();
    expect(within(telemetry!).getByText("Maximum attempts")).toBeVisible();
    expect(within(telemetry!).getByText("Maximum pending spans")).toBeVisible();
    expect(within(telemetry!).getByText("Initial backoff")).toBeVisible();
    expect(within(telemetry!).getByText("100 ms")).toBeVisible();
    expect(within(telemetry!).getByText("Maximum backoff")).toBeVisible();
    expect(within(telemetry!).getByText("1000 ms")).toBeVisible();
    expect(telemetry).not.toHaveTextContent("[object Object]");
    const gateway = screen
      .getByRole("heading", { name: "Worker LLM Gateway" })
      .closest("article");
    expect(gateway).not.toBeNull();
    expect(within(gateway!).getByText("Gateway ID")).toBeVisible();
    expect(within(gateway!).getByText("local-litellm")).toBeVisible();
    expect(within(gateway!).getByText("Credential")).toBeVisible();
    expect(within(gateway!).getByText("Explicitly cleared")).toBeVisible();
    expect(gateway).not.toHaveTextContent("null");
  });

  it("confirms permanent Runtime credential deletion after safe dismissals", async () => {
    const { deletes } = setup("/operations/configuration", true, true);
    const user = userEvent.setup();
    const trigger = await screen.findByRole("button", {
      name: "Delete Runtime credential caido-local",
    });
    const open = async () => {
      await user.click(trigger);
      const dialog = screen.getByRole("alertdialog", {
        name: "Delete Runtime credential caido-local?",
      });
      expect(
        within(dialog).getByRole("button", { name: "Cancel" }),
      ).toHaveFocus();
      expect(dialog).toHaveTextContent("Its ID cannot be reused");
      expect(dialog).toHaveTextContent("cannot be bound again");
      expect(deletes).toHaveLength(0);
      return dialog;
    };
    let dialog = await open();
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alertdialog")).toBeNull();
    await open();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog")).toBeNull();
    dialog = await open();
    await user.click(dialog.parentElement!);
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(deletes).toHaveLength(0);
    expect(trigger).toBeInTheDocument();
    dialog = await open();
    await user.click(
      within(dialog).getByRole("button", { name: "Delete Runtime credential" }),
    );
    await waitFor(() => expect(deletes).toHaveLength(1));
    expect(new URL(deletes[0]!.url).pathname).toBe(
      "/v1/operations/runtime-credentials/caido-local",
    );
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    await waitFor(() => expect(trigger).not.toBeInTheDocument());
  });

  it("shows every Runtime credential deletion blocker in the dialog", async () => {
    setup("/operations/configuration", true, true, undefined, {
      path: "/v1/operations/runtime-credentials/caido-local",
      body: {
        code: "runtime_credential_in_use",
        message: "Runtime credential is referenced by active configuration",
        retryable: false,
        details: {
          kind: "runtime_credential_in_use",
          bindingLabels: ["debug"],
          projectIds: ["project-one"],
          runIds: ["run-one"],
          auditIds: ["audit-one"],
          allocationIds: ["allocation-one"],
        },
      },
    });
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", {
        name: "Delete Runtime credential caido-local",
      }),
    );
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete Runtime credential caido-local?",
    });
    await user.click(
      within(dialog).getByRole("button", { name: "Delete Runtime credential" }),
    );
    expect(await within(dialog).findByText("audit-one")).toBeVisible();
    expect(within(dialog).getByRole("link", { name: "debug" })).toHaveAttribute(
      "href",
      "/operations/configuration",
    );
    expect(
      within(dialog).getByRole("link", { name: "project-one" }),
    ).toHaveAttribute("href", "/projects/project-one");
    expect(
      within(dialog).getByRole("link", { name: "run-one" }),
    ).toHaveAttribute("href", "/runs/run-one");
    expect(within(dialog).getByText("allocation-one")).toBeVisible();
  });

  it("confirms Runtime label removal after safe dismissals", async () => {
    const { deletes } = setup("/operations/configuration", true, true);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", {
        name: "Manage bindings for debug@1",
      }),
    );
    const trigger = screen.getByRole("button", {
      name: "Remove binding debug",
    });
    const open = async () => {
      await user.click(trigger);
      const dialog = screen.getByRole("alertdialog", {
        name: "Remove binding debug?",
      });
      expect(
        within(dialog).getByRole("button", { name: "Cancel" }),
      ).toHaveFocus();
      expect(dialog).toHaveTextContent(
        "New Runs will no longer be able to select",
      );
      expect(deletes).toHaveLength(0);
      return dialog;
    };
    let dialog = await open();
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await open();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("alertdialog")).toBeNull();
    dialog = await open();
    await user.click(dialog.parentElement!);
    expect(screen.queryByRole("alertdialog")).toBeNull();
    expect(deletes).toHaveLength(0);
    expect(trigger).toBeInTheDocument();
    dialog = await open();
    await user.click(
      within(dialog).getByRole("button", { name: "Remove binding" }),
    );
    await waitFor(() => expect(deletes).toHaveLength(1));
    expect(new URL(deletes[0]!.url).pathname).toBe(
      "/v1/operations/runtime-labels/debug",
    );
    expect(deletes[0]!.headers.get("If-Match")).toBe('"1"');
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    await waitFor(() => expect(trigger).not.toBeInTheDocument());
  });

  it("shows Runtime Agent IDs blocking label removal", async () => {
    const agentID = "a".repeat(64);
    setup("/operations/configuration", true, true, undefined, {
      path: "/v1/operations/runtime-labels/debug",
      body: {
        code: "runtime_label_in_use",
        message: "Runtime label is assigned to a Runtime Agent",
        retryable: false,
        details: {
          kind: "runtime_label_in_use",
          runtimeAgentIds: [agentID],
        },
      },
    });
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", {
        name: "Manage bindings for debug@1",
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Remove binding debug" }),
    );
    const dialog = screen.getByRole("alertdialog", {
      name: "Remove binding debug?",
    });
    await user.click(
      within(dialog).getByRole("button", { name: "Remove binding" }),
    );
    expect(await within(dialog).findByText(agentID)).toBeVisible();
    expect(
      within(dialog).getByRole("heading", { name: "Runtime Agents" }),
    ).toBeVisible();
  });

  it("opens creation forms from icons and clears a closed credential draft", async () => {
    setup("/operations/configuration");
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

  it("opens Runtime configuration as an Operations section with a scope label and refresh", async () => {
    const { router, requests } = setup("/operations/configuration");
    await screen.findByRole("heading", { name: "RuntimeConfig versions" });
    expect(await screen.findByRole("link", { name: "debug@1" })).toBeVisible();
    expect(router.state.location.pathname).toBe("/operations/configuration");
    expect(
      screen.getByRole("heading", { name: "Runtime configuration" }),
    ).toBeVisible();
    expect(screen.getByText("Server-wide")).toHaveClass("state-badge");
    const tabs = screen.getByRole("navigation", {
      name: "Operations sections",
    });
    expect(
      within(tabs)
        .getAllByRole("link")
        .map((link) => link.textContent?.trim()),
    ).toEqual([
      "Overview",
      "Runtime Agents",
      "Allocations",
      "Performance",
      "Configuration",
      "LLM configurations",
      "Credentials",
      "Settings",
    ]);
    expect(tabs).toHaveTextContent("Setup");
    expect(
      within(tabs).getByRole("link", { name: "Configuration" }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(tabs).getByRole("link", { name: "LLM configurations" }),
    ).not.toHaveAttribute("aria-current");
    expect(
      within(
        screen.getByRole("combobox", { name: "Operations section" }),
      ).getByRole("option", { name: "Configuration" }),
    ).toHaveValue("/operations/configuration");
    expect(screen.queryByRole("navigation", { name: "Run views" })).toBeNull();
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
      "/operations/configuration/debug/1?from=bookmark#worker",
    );
    await screen.findByRole("heading", { name: "debug@1" });
    expect(screen.getByRole("heading", { name: "Worker Caido" })).toBeVisible();
    expect(screen.getByText("https://caido.example/graphql")).toBeVisible();
    expect(router.state.location).toMatchObject({
      pathname: "/operations/configuration/debug/1",
      search: "?from=bookmark",
      hash: "#worker",
    });
    expect(
      screen.getByRole("link", { name: /← Runtime configuration/ }),
    ).toHaveAttribute("href", "/operations/configuration");
  });

  it("keeps no redirect from the removed Runs configuration location", async () => {
    const { router } = setup(
      "/runs/configuration/debug/1?from=bookmark#worker",
    );
    expect(
      await screen.findByRole("heading", { name: "Page not found" }),
    ).toBeVisible();
    expect(router.state.location).toMatchObject({
      pathname: "/runs/configuration/debug/1",
      search: "?from=bookmark",
      hash: "#worker",
    });
    expect(router.state.historyAction).toBe("POP");
  });

  it("drops the Configuration tab from Runs", async () => {
    setup("/runs");
    const tabs = await screen.findByRole("navigation", { name: "Run views" });
    expect(
      within(tabs)
        .getAllByRole("link")
        .map((link) => link.textContent?.trim()),
    ).toEqual(["Queue", "Completed"]);
  });

  it.each(["/operations/configuration", "/operations/configuration/debug/1"])(
    "keeps operator authorization on %s",
    async (path) => {
      const { requests } = setup(path, false);
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "not authorized to observe or manage Operations",
      );
      expect(screen.queryByRole("link", { name: /Configuration/ })).toBeNull();
      expect(
        requests.filter((request) => !request.startsWith("/v1/projects")),
      ).toEqual(["/v1/auth/session"]);
    },
  );
});
