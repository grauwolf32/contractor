import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { components } from "../../../api/generated/public";
import { renderApplication } from "../../../test/application";
import { operationsSnapshot } from "../../../test/builders-execution";
import { conforming } from "../../../test/builders";
import { json, noContent, page } from "../../../test/fake-server";

type Schemas = components["schemas"];
type Worker = Schemas["RuntimeConfigDocument"]["spec"]["worker"];

const CREATED = "2026-09-07T00:00:00Z";

function runtimeConfig(worker?: Worker): Schemas["RuntimeConfigResource"] {
  return conforming("RuntimeConfigResource", {
    ref: { name: "debug", version: "1", digest: `sha256:${"1".repeat(64)}` },
    document: {
      apiVersion: "contractor/v1alpha1",
      kind: "RuntimeConfig",
      metadata: { name: "debug", version: "1" },
      spec: {
        worker: worker ?? {
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
    createdAt: CREATED,
  });
}

/** A 409 refusal whose details name what still uses the resource. */
function inUse(
  code: string,
  message: string,
  details: NonNullable<Schemas["Error"]["details"]>,
): Response {
  return json(
    conforming("Error", {
      code,
      message,
      retryable: false,
      requestId: "request_test",
      details,
    }),
    409,
  );
}

/**
 * Operations at `path`, holding one RuntimeConfig version and, with
 * `inventory`, one credential and one label bound to it. `refusal` answers
 * deletions instead of removing the resource.
 */
function setup(
  path: string,
  options: {
    authorized?: boolean;
    inventory?: boolean;
    worker?: Worker;
    refusal?: () => Response;
  } = {},
) {
  const resource = runtimeConfig(options.worker);
  let credentials = options.inventory
    ? [
        conforming("RuntimeCredentialMetadata", {
          credentialId: "caido-local",
          kind: "caido-bearer@1",
          createdBy: "user_local",
          createdAt: CREATED,
        }),
      ]
    : [];
  let bindings = options.inventory
    ? [
        conforming("RuntimeLabelBinding", {
          label: "debug",
          config: resource.ref,
          revision: "1",
          createdBy: "user_local",
          createdAt: CREATED,
          updatedBy: "user_local",
          updatedAt: CREATED,
        }),
      ]
    : [];
  const view = renderApplication(path, {
    capabilities:
      options.authorized === false ? ["user"] : ["user", "operations"],
    server: {
      "GET /v1/operations/snapshot": () => json(operationsSnapshot("1")),
      "GET /v1/operations/runtime-configs": () => json(page([resource])),
      "GET /v1/operations/runtime-configs/{name}/versions/{version}": () =>
        json(resource),
      "GET /v1/operations/runtime-credentials": () => json(page(credentials)),
      "GET /v1/operations/runtime-labels": () => json(page(bindings)),
      "DELETE /v1/operations/runtime-credentials/{credentialId}": () => {
        if (options.refusal !== undefined) return options.refusal();
        credentials = [];
        return noContent();
      },
      "DELETE /v1/operations/runtime-labels/{label}": () => {
        if (options.refusal !== undefined) return options.refusal();
        bindings = [];
        return noContent();
      },
    },
  });
  return {
    ...view,
    deletes: () =>
      view.requests.filter((request) => request.method === "DELETE"),
    paths: () => view.requests.map((request) => new URL(request.url).pathname),
  };
}

describe("Runtime configuration hub navigation", () => {
  it("shows nested RuntimeConfig export settings and explicit clears with labels", async () => {
    setup("/operations/configuration/debug/1", {
      worker: {
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
      },
    });
    const telemetry = (
      await screen.findByRole("heading", { name: "Worker telemetry" })
    ).closest("article")!;
    for (const text of [
      "Batch size",
      "8.0 MiB",
      "Maximum pending bytes",
      "64.0 MiB",
      "Maximum attempts",
      "Maximum pending spans",
      "Initial backoff",
      "100 ms",
      "Maximum backoff",
      "1000 ms",
    ]) {
      expect(within(telemetry).getByText(text)).toBeVisible();
    }
    expect(telemetry).not.toHaveTextContent("[object Object]");
    const gateway = screen
      .getByRole("heading", { name: "Worker LLM Gateway" })
      .closest("article")!;
    for (const text of [
      "Gateway ID",
      "local-litellm",
      "Credential",
      "Explicitly cleared",
    ]) {
      expect(within(gateway).getByText(text)).toBeVisible();
    }
    expect(gateway).not.toHaveTextContent("null");
  });

  it("confirms permanent Runtime credential deletion after safe dismissals", async () => {
    const { deletes, user } = setup("/operations/configuration", {
      inventory: true,
    });
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
      expect(deletes()).toHaveLength(0);
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
    expect(deletes()).toHaveLength(0);
    expect(trigger).toBeInTheDocument();
    dialog = await open();
    await user.click(
      within(dialog).getByRole("button", { name: "Delete Runtime credential" }),
    );
    await waitFor(() => expect(deletes()).toHaveLength(1));
    expect(new URL(deletes()[0]!.url).pathname).toBe(
      "/v1/operations/runtime-credentials/caido-local",
    );
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    await waitFor(() => expect(trigger).not.toBeInTheDocument());
  });

  it("shows every Runtime credential deletion blocker in the dialog", async () => {
    const { user } = setup("/operations/configuration", {
      inventory: true,
      refusal: () =>
        inUse(
          "runtime_credential_in_use",
          "Runtime credential is referenced by active configuration",
          {
            kind: "runtime_credential_in_use",
            bindingLabels: ["debug"],
            projectIds: ["project-one"],
            runIds: ["run-one"],
            auditIds: ["audit-one"],
            allocationIds: ["allocation-one"],
          },
        ),
    });
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
    for (const [name, href] of [
      ["debug", "/operations/configuration"],
      ["project-one", "/projects/project-one"],
      ["run-one", "/runs/run-one"],
    ] as const) {
      expect(within(dialog).getByRole("link", { name })).toHaveAttribute(
        "href",
        href,
      );
    }
    expect(within(dialog).getByText("allocation-one")).toBeVisible();
  });

  it("confirms Runtime label removal after safe dismissals", async () => {
    const { deletes, user } = setup("/operations/configuration", {
      inventory: true,
    });
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
      expect(deletes()).toHaveLength(0);
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
    expect(deletes()).toHaveLength(0);
    expect(trigger).toBeInTheDocument();
    dialog = await open();
    await user.click(
      within(dialog).getByRole("button", { name: "Remove binding" }),
    );
    await waitFor(() => expect(deletes()).toHaveLength(1));
    expect(new URL(deletes()[0]!.url).pathname).toBe(
      "/v1/operations/runtime-labels/debug",
    );
    expect(deletes()[0]!.headers.get("If-Match")).toBe('"1"');
    await waitFor(() => expect(screen.queryByRole("alertdialog")).toBeNull());
    await waitFor(() => expect(trigger).not.toBeInTheDocument());
  });

  it("shows Runtime Agent IDs blocking label removal", async () => {
    const agentID = "a".repeat(64);
    const { user } = setup("/operations/configuration", {
      inventory: true,
      refusal: () =>
        inUse(
          "runtime_label_in_use",
          "Runtime label is assigned to a Runtime Agent",
          { kind: "runtime_label_in_use", runtimeAgentIds: [agentID] },
        ),
    });
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
    const { user } = setup("/operations/configuration");
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
    const { router, paths, user } = setup("/operations/configuration");
    await screen.findByRole("heading", { name: "RuntimeConfig versions" });
    expect(await screen.findByRole("link", { name: "debug@1" })).toBeVisible();
    expect(router.state.location.pathname).toBe("/operations/configuration");
    expect(
      screen.getByRole("heading", { name: "Runtime configuration" }),
    ).toBeVisible();
    expect(screen.getByText("Server-wide")).toHaveClass("ops-scope");
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
    const reads = () =>
      paths().filter((path) => path === "/v1/operations/runtime-configs")
        .length;
    const before = reads();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(reads()).toBeGreaterThan(before));
  });

  it("opens exact versions with query and fragment", async () => {
    const { router } = setup(
      "/operations/configuration/debug/1?from=bookmark#worker",
    );
    await screen.findByRole("heading", { name: "debug@1" });
    expect(
      await screen.findByRole("heading", { name: "Worker Caido" }),
    ).toBeVisible();
    expect(screen.getByText("https://caido.example/graphql")).toBeVisible();
    expect(router.state.location).toMatchObject({
      pathname: "/operations/configuration/debug/1",
      search: "?from=bookmark",
      hash: "#worker",
    });
    expect(
      within(screen.getByRole("navigation", { name: "Breadcrumb" })).getByRole(
        "link",
        { name: "Runtime configuration" },
      ),
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
      const { paths } = setup(path, { authorized: false });
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "not authorized to observe or manage Operations",
      );
      expect(screen.queryByRole("link", { name: /Configuration/ })).toBeNull();
      expect(
        paths().filter(
          (request) =>
            ![
              "/v1/projects",
              "/v1/audits",
              "/v1/findings",
              "/v1/reviews",
            ].includes(request),
        ),
      ).toEqual(["/v1/auth/session"]);
    },
  );
});
