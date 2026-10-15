import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type {
  ConfigurationResource,
  CreateCredentialRequest,
  RuntimeConfigAuthorDocument,
  RuntimeCredentialMetadata,
} from "../../api/operations";
import {
  emptyPageOrNotFound,
  failure,
  json,
  noContent,
  page,
} from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { CredentialCreateForm } from "./credentials/form";
import { RuntimeConfigurationRoute } from "./runtime-configs/index";

const now = "2026-09-19T10:00:00Z";

function digest(index: number): string {
  return `sha256:${index.toString(16).padStart(64, "0")}`;
}

function gateway(index: number, managed = false): ConfigurationResource {
  return {
    ref: {
      kind: "llm-gateways",
      name: `gateway-${String(index).padStart(2, "0")}`,
      version: "1",
      digest: digest(index + 1),
    },
    source: "managed",
    body: {
      protocol: "openai-compatible@1",
      url: "http://127.0.0.1:4000/v1/",
      ...(managed
        ? {
            credentialManager: {
              implementation: "litellm-virtual-keys@1" as const,
              managementUrl: "http://127.0.0.1:4000/",
            },
          }
        : {}),
    },
  };
}

function policy(index: number): ConfigurationResource {
  return {
    ref: {
      kind: "model-policies",
      name: `policy-${String(index).padStart(2, "0")}`,
      version: "1",
      digest: digest(index + 100),
    },
    source: "managed",
    body: { model: `model-${index}` },
  };
}

function credential(index: number): RuntimeCredentialMetadata {
  return {
    credentialId: `credential-${String(index).padStart(2, "0")}`,
    kind: "caido-bearer@1",
    createdBy: "operator",
    createdAt: now,
  };
}

/** Two pages of 50 and the rest; `cursor` is null for the first. */
function split<T>(items: T[], cursor: string | null) {
  return cursor === null
    ? page(items.slice(0, 50), "page-2")
    : page(items.slice(50));
}

describe("Operations inventories beyond the first page", () => {
  it("submits a managed Gateway and ModelPolicy from later pages", async () => {
    const gateways = Array.from({ length: 60 }, (_, index) =>
      gateway(index, index === 51),
    );
    const policies = Array.from({ length: 60 }, (_, index) => policy(index));
    const posts: CreateCredentialRequest[] = [];
    renderWithAPI(<CredentialCreateForm />, {
      server: async (request, url) => {
        const cursor = url.searchParams.get("cursor");
        if (url.pathname === "/v1/configurations/llm-gateways")
          return json(split(gateways, cursor));
        if (url.pathname === "/v1/configurations/model-policies")
          return json(split(policies, cursor));
        if (
          url.pathname === "/v1/operations/credentials" &&
          request.method === "POST"
        ) {
          const body = (await request.json()) as CreateCredentialRequest;
          posts.push(body);
          return json(
            {
              credentialId: body.credentialId,
              llmGateway: body.llmGateway,
              createdAt: now,
              effectivePolicy: {
                modelPolicies: body.gatewayPolicy.modelPolicies,
                models: ["model-59"],
              },
            },
            201,
          );
        }
        return emptyPageOrNotFound(request, url);
      },
    });
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Load more Gateways" }),
    );
    await screen.findByRole("option", { name: /gateway-51@1/ });
    await user.click(
      await screen.findByRole("button", { name: "Load more ModelPolicies" }),
    );
    await screen.findByRole("checkbox", { name: /policy-59@1/ });
    await user.type(
      screen.getByRole("textbox", { name: "Credential ID" }),
      "late-credential",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Managed LLM Gateway" }),
      `${gateways[51]!.ref.name}@1:${gateways[51]!.ref.digest}`,
    );
    await user.click(screen.getByRole("checkbox", { name: /policy-59@1/ }));
    await user.click(
      screen.getByRole("button", { name: "Create active credential" }),
    );
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]?.llmGateway.gatewayId).toBe("gateway-51");
    expect(posts[0]?.gatewayPolicy.modelPolicies).toEqual([
      { policyId: "policy-59", version: "1", digest: policies[59]!.ref.digest },
    ]);
  });

  it("publishes a RuntimeConfig with a Gateway from page two", async () => {
    const gateways = Array.from({ length: 60 }, (_, index) => gateway(index));
    const posts: RuntimeConfigAuthorDocument[] = [];
    renderWithAPI(<RuntimeConfigurationRoute />, {
      server: async (request, url) => {
        if (url.pathname === "/v1/configurations/llm-gateways")
          return json(split(gateways, url.searchParams.get("cursor")));
        if (
          url.pathname === "/v1/operations/runtime-configs" &&
          request.method === "POST"
        ) {
          const document =
            (await request.json()) as RuntimeConfigAuthorDocument;
          posts.push(document);
          // The Server answers with the exact Gateway the selector resolved to.
          const selected = gateways.find(
            (candidate) =>
              `${candidate.ref.name}@${candidate.ref.version}` ===
              document.spec.worker?.llmGateway?.gateway,
          );
          return json(
            {
              ref: {
                name: document.metadata.name,
                version: document.metadata.version,
                digest: digest(900),
              },
              document: {
                ...document,
                spec: {
                  worker: {
                    llmGateway: {
                      gateway: {
                        gatewayId: selected?.ref.name,
                        version: selected?.ref.version,
                        digest: selected?.ref.digest,
                      },
                    },
                  },
                },
              },
              builtIn: false,
              createdBy: "operator",
              createdAt: now,
            },
            201,
          );
        }
        return emptyPageOrNotFound(request, url);
      },
    });
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Publish RuntimeConfig" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Publish RuntimeConfig",
    });
    await user.click(
      await within(dialog).findByRole("button", { name: "Load more Gateways" }),
    );
    await within(dialog).findByRole("option", { name: /gateway-51@1/ });
    await user.type(
      within(dialog).getByRole("textbox", { name: "RuntimeConfig name" }),
      "later-gateway",
    );
    await user.click(
      within(dialog).getByRole("checkbox", {
        name: /Worker LLM Gateway route/,
      }),
    );
    await user.selectOptions(
      within(dialog).getByRole("combobox", { name: "LLM Gateway" }),
      `${gateways[51]!.ref.name}@1:${gateways[51]!.ref.digest}`,
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Publish RuntimeConfig" }),
    );
    await waitFor(() => expect(posts).toHaveLength(1));
    expect(posts[0]?.spec.worker?.llmGateway?.gateway).toBe("gateway-51@1");
  });

  it("shows and deletes a Runtime credential after page one", async () => {
    let credentials = Array.from({ length: 60 }, (_, index) =>
      credential(index),
    );
    const deletes: string[] = [];
    renderWithAPI(<RuntimeConfigurationRoute />, {
      server: async (request, url) => {
        if (
          url.pathname === "/v1/operations/runtime-credentials" &&
          request.method === "GET"
        )
          return json(split(credentials, url.searchParams.get("cursor")));
        if (
          url.pathname === "/v1/operations/runtime-credentials/credential-59" &&
          request.method === "DELETE"
        ) {
          deletes.push(url.pathname);
          credentials = credentials.filter(
            (item) => item.credentialId !== "credential-59",
          );
          return noContent();
        }
        return emptyPageOrNotFound(request, url);
      },
    });
    const user = userEvent.setup();
    const navigation = await screen.findByRole("navigation", {
      name: "Runtime credential pages",
    });
    await user.click(within(navigation).getByRole("button", { name: "Next" }));
    await user.click(
      await screen.findByRole("button", {
        name: "Delete Runtime credential credential-59",
      }),
    );
    const dialog = screen.getByRole("alertdialog", {
      name: "Delete Runtime credential credential-59?",
    });
    await user.click(
      within(dialog).getByRole("button", { name: "Delete Runtime credential" }),
    );
    await waitFor(() =>
      expect(deletes).toEqual([
        "/v1/operations/runtime-credentials/credential-59",
      ]),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("button", {
          name: "Delete Runtime credential credential-59",
        }),
      ).not.toBeInTheDocument(),
    );
  });

  it("does not report an empty inventory while a page loads or fails", async () => {
    let credentialsAvailable = false;
    renderWithAPI(<RuntimeConfigurationRoute />, {
      server: async (request, url) => {
        if (url.pathname === "/v1/operations/runtime-configs")
          return new Promise<Response>(() => undefined);
        if (url.pathname === "/v1/operations/runtime-credentials")
          return credentialsAvailable
            ? json(page([credential(1)]))
            : failure(
                503,
                "unavailable",
                "Runtime credentials are unavailable",
                true,
              );
        return emptyPageOrNotFound(request, url);
      },
    });
    expect(await screen.findByText("Loading RuntimeConfigs…")).toBeVisible();
    expect(
      await screen.findByText("Could not load Runtime credentials"),
    ).toBeVisible();
    expect(
      screen.queryByText("No RuntimeConfig versions are visible."),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("No Runtime credentials exist."),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/on this page/)).not.toBeInTheDocument();

    credentialsAvailable = true;
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("credential-01")).toBeVisible();
    expect(screen.getByText("1 on this page")).toBeVisible();
  });

  it("labels an empty later page without claiming the inventory is empty", async () => {
    const config = {
      ref: { name: "debug", version: "1", digest: digest(500) },
      document: {
        apiVersion: "contractor/v1alpha1",
        kind: "RuntimeConfig",
        metadata: { name: "debug", version: "1" },
        spec: {},
      },
      builtIn: false,
      createdBy: "operator",
      createdAt: now,
    };
    const firstOrEmpty = (item: unknown, cursor: string | null) =>
      cursor === null
        ? { items: [item], page: { hasMore: true, nextCursor: "page-2" } }
        : { items: [], page: { hasMore: false } };
    renderWithAPI(<RuntimeConfigurationRoute />, {
      server: async (request, url) => {
        const cursor = url.searchParams.get("cursor");
        if (url.pathname === "/v1/operations/runtime-configs")
          return json(firstOrEmpty(config, cursor));
        if (url.pathname === "/v1/operations/runtime-credentials")
          return json(firstOrEmpty(credential(1), cursor));
        return emptyPageOrNotFound(request, url);
      },
    });
    const user = userEvent.setup();
    for (const label of ["RuntimeConfig pages", "Runtime credential pages"]) {
      const navigation = await screen.findByRole("navigation", { name: label });
      await user.click(
        await within(navigation).findByRole("button", { name: "Next" }),
      );
    }
    expect(
      await screen.findByText(
        "This page lists no RuntimeConfig versions. Earlier pages may list more.",
      ),
    ).toBeVisible();
    expect(
      await screen.findByText(
        "This page lists no Runtime credentials. Earlier pages may list more.",
      ),
    ).toBeVisible();
    expect(
      screen.queryByText("No RuntimeConfig versions are visible."),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText("No Runtime credentials exist."),
    ).not.toBeInTheDocument();
  });
});
