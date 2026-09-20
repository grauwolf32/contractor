import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import type {
  RuntimeConfigAuthorDocument,
  RuntimeLabelBinding,
} from "../../api/operations";
import type { WorkflowResource } from "../../api/workflows";
import { createApplicationQueryClient } from "../../app/query-client";
import { RunDraftStoreContext } from "../../run-drafts/context";
import {
  initialRunDraftState,
  RunDraftMemoryStore,
} from "../../run-drafts/memory";
import {
  resolveToolsetCoverage,
  ToolsetConnectionCreation,
} from "../../run-drafts/toolset-connections";
import {
  MCPConnectionDialog,
  ToolsetConfigurationRequirements,
} from "./toolset-configurations";
import { WorkflowRunForm } from "./run-form";

const digest = `sha256:${"1".repeat(64)}`;
const otherDigest = `sha256:${"2".repeat(64)}`;
const timestamp = "2026-09-20T12:00:00Z";
const config = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
const requirements: NonNullable<
  WorkflowResource["toolsetConfigurationRequirements"]
> = [
  {
    ref: "mcp@1",
    configurationRequired: true,
    configurationSchema: "mcp-streamable-http@1",
    toolDiscovery: "allocation",
    workerRuntimes: ["adk@1"],
    tools: ["fetch", "search"],
    consumers: [
      { stage: "later", agent: "reader" },
      { stage: "start", agent: "researcher" },
    ],
  },
];
function binding(
  label: string,
  value: string | null | undefined = digest,
): RuntimeLabelBinding {
  return {
    label,
    config: {
      name: label === "default" ? "empty" : label,
      version: "1",
      digest,
    },
    revision: "1",
    createdBy: "owner",
    createdAt: timestamp,
    updatedBy: "owner",
    updatedAt: timestamp,
    ...(value === undefined
      ? {}
      : {
          toolsetConfigurationCoverage: [
            value === null
              ? { ref: "mcp@1", state: "cleared" }
              : { ref: "mcp@1", state: "configured", valueDigest: value },
          ],
          changedSections: ["worker.llmGateway", "worker.toolsets"],
        }),
  };
}
const emptyDefault: RuntimeLabelBinding = {
  label: "default",
  config: { name: "empty", version: "1", digest },
  revision: "1",
  createdBy: "owner",
  createdAt: timestamp,
  updatedBy: "owner",
  updatedAt: timestamp,
};
function response(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
      ETag: '"1"',
    },
  });
}
function document(): RuntimeConfigAuthorDocument {
  return {
    apiVersion: "contractor/v1alpha1",
    kind: "RuntimeConfig",
    metadata: { name: "knowledge", version: "1" },
    spec: {
      worker: {
        toolsets: {
          "mcp@1": {
            endpoint: "https://private-tools.example/mcp",
            connectTimeoutSeconds: 10,
            requestTimeoutSeconds: 60,
          },
        },
      },
    },
  };
}
function credential() {
  return {
    credentialId: "knowledge-secret",
    kind: "mcp-headers@1" as const,
    material: { headers: { authorization: "Bearer private-test-sentinel" } },
  };
}
function service(
  failAt?: "credential" | "configuration" | "label",
  rejected = false,
) {
  const commits = new Map<string, unknown>();
  const calls: { step: string; key: string; body: unknown }[] = [];
  let fault = true;
  const api = new PublicAPI(
    config,
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const path = new URL(request.url).pathname;
      if (request.method === "GET")
        return response({ items: [], page: { hasMore: false } });
      const body = await request.json();
      const step = path.endsWith("runtime-credentials")
        ? "credential"
        : path.endsWith("runtime-configs")
          ? "configuration"
          : "label";
      const key = request.headers.get("Idempotency-Key")!;
      calls.push({ step, key, body });
      if (step === "label") {
        expect(request.headers.get("If-None-Match")).toBe("*");
        expect(request.headers.has("If-Match")).toBe(false);
      }
      if (step === failAt && fault && rejected) {
        fault = false;
        return response(
          {
            code: "unavailable",
            message: "Rejected before commit",
            retryable: true,
          },
          503,
        );
      }
      if (!commits.has(key))
        commits.set(
          key,
          step === "credential"
            ? {
                credentialId: body.credentialId,
                kind: body.kind,
                createdBy: "owner",
                createdAt: timestamp,
              }
            : step === "configuration"
              ? {
                  ref: {
                    name: body.metadata.name,
                    version: body.metadata.version,
                    digest,
                  },
                  document: body,
                  builtIn: false,
                  createdBy: "owner",
                  createdAt: timestamp,
                }
              : { ...binding(path.split("/").at(-1)!), config: body.config },
        );
      if (step === failAt && fault) {
        fault = false;
        throw new TypeError("reply lost after commit");
      }
      return response(commits.get(key), 201);
    }),
  );
  api.csrf.replace("test-csrf");
  return { api, calls, commits };
}
function dialog(
  api: PublicAPI,
  progress: ToolsetConnectionCreation,
  onCreated = vi.fn(),
) {
  const client = createApplicationQueryClient();
  return {
    ...render(
      <QueryClientProvider client={client}>
        <PublicAPIProvider api={api}>
          <MemoryRouter>
            <MCPConnectionDialog
              progress={progress}
              onStarted={() => true}
              onCreated={onCreated}
              onClose={vi.fn()}
            />
          </MemoryRouter>
        </PublicAPIProvider>
      </QueryClientProvider>,
    ),
    client,
    onCreated,
  };
}

const workflow: WorkflowResource = {
  ref: { name: "research", version: "1" },
  entryStage: "start",
  parameters: {},
  inputs: {},
  outputs: {},
  stages: {},
  toolsetConfigurationRequirements: requirements,
};
function form(
  labels: RuntimeLabelBinding[],
  selected: string[] = [],
  inventory = labels,
) {
  const paths: string[] = [];
  const runs: unknown[] = [];
  const api = new PublicAPI(
    config,
    vi.fn(async (input) => {
      const request = input instanceof Request ? input : new Request(input);
      const path = new URL(request.url).pathname;
      paths.push(path);
      if (path === "/v1/runs") {
        runs.push(await request.json());
        return response(
          { code: "unavailable", message: "Run fixture stop", retryable: true },
          503,
        );
      }
      if (path.startsWith("/v1/operations/runtime-labels/")) {
        const row = labels.find(
          (item) => item.label === path.split("/").at(-1),
        );
        return row
          ? response(row)
          : response(
              {
                code: "not_found",
                message: "Label was deleted",
                retryable: false,
              },
              404,
            );
      }
      return response({
        items: path === "/v1/operations/runtime-labels" ? inventory : [],
        page: { hasMore: false },
      });
    }),
  );
  api.csrf.replace("test-csrf");
  const store = new RunDraftMemoryStore("owner");
  const state = initialRunDraftState();
  state.runtimeLabels = selected;
  const entry = store.acquire(
    { workflowName: "research", workflowVersion: "1" },
    state,
  );
  if (entry.kind !== "acquired") throw new Error("draft unavailable");
  const client = createApplicationQueryClient();
  render(
    <QueryClientProvider client={client}>
      <PublicAPIProvider api={api}>
        <RunDraftStoreContext value={store}>
          <MemoryRouter>
            <WorkflowRunForm workflow={workflow} />
          </MemoryRouter>
        </RunDraftStoreContext>
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return { paths, runs, store, entry: entry.entry, client };
}

describe("Toolset profile coverage", () => {
  it.each([
    {
      selected: [],
      labels: [binding("default")],
      status: "Inherited from default",
    },
    {
      selected: [],
      labels: [emptyDefault],
      status: "Choose a connection profile",
    },
    {
      selected: ["clear"],
      labels: [binding("default"), binding("clear", null)],
      status: "Cleared by clear",
    },
    {
      selected: ["a", "b"],
      labels: [emptyDefault, binding("a"), binding("b")],
      status: "Configured by a, b",
    },
    {
      selected: ["a", "b"],
      labels: [emptyDefault, binding("a"), binding("b", otherDigest)],
      status: "Conflicting profiles: a, b",
    },
  ])("shows $status using safe projections", ({ selected, labels, status }) => {
    render(
      <ToolsetConfigurationRequirements
        requirements={requirements}
        coverage={resolveToolsetCoverage(["mcp@1"], selected, labels)}
        onCreate={vi.fn()}
      />,
    );
    expect(screen.getByText(status)).toBeVisible();
    expect(
      screen.getByText("Used by later/reader, start/researcher"),
    ).toBeVisible();
  });
  it("selects one mixed profile for two consumers and submits only runtimeLabels", async () => {
    const { paths, runs } = form([emptyDefault, binding("knowledge")]);
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("checkbox", { name: /knowledge/ }),
    );
    expect(await screen.findByText("Configured by knowledge")).toBeVisible();
    expect(
      screen.getByText(/Changes: worker.llmGateway, worker.toolsets/),
    ).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await waitFor(() => expect(runs).toHaveLength(1));
    expect(runs[0]).toEqual({
      workflow: "research@1",
      runtimeLabels: ["knowledge"],
      parameters: {},
      artifacts: {},
    });
    expect(
      paths.some((path) => path.startsWith("/v1/operations/runtime-configs")),
    ).toBe(false);
  });
  it("retains selected labels outside the loaded page and rejects deleted labels", async () => {
    const { runs } = form(
      [emptyDefault, binding("outside")],
      ["outside", "deleted"],
      [emptyDefault],
    );
    expect(
      await screen.findByText(/Loading or unavailable profiles: deleted/),
    ).toBeVisible();
    expect(screen.getByRole("checkbox", { name: /outside/ })).toBeChecked();
    await userEvent.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    expect(runs).toHaveLength(0);
    expect(await screen.findByText(/Resolve missing, cleared/)).toBeVisible();
  });
});

describe("Connection creation recovery", () => {
  it.each(["credential", "configuration", "label"] as const)(
    "replays a lost %s reply without duplicate commits, including dialog navigation",
    async (step) => {
      const storageWrite = vi.spyOn(Storage.prototype, "setItem");
      const { api, calls, commits } = service(step);
      const progress = new ToolsetConnectionCreation();
      const store = new RunDraftMemoryStore("owner");
      const id = { workflowName: "research", workflowVersion: "1" };
      const acquired = store.acquire(id, initialRunDraftState());
      if (acquired.kind !== "acquired") throw new Error("draft unavailable");
      store.keepConnectionCreation(acquired.entry, progress);
      store.retain(acquired.entry);
      const first = dialog(api, progress);
      const user = userEvent.setup();
      await user.type(screen.getByLabelText("Connection name"), "knowledge");
      await user.type(
        screen.getByLabelText("MCP endpoint"),
        document().spec.worker!.toolsets!["mcp@1"]!.endpoint,
      );
      await user.selectOptions(screen.getByLabelText("Authentication"), "new");
      await user.type(
        screen.getByLabelText("New MCP credential ID"),
        "knowledge-secret",
      );
      await user.type(
        screen.getByLabelText("Header value · write only"),
        "Bearer private-test-sentinel",
      );
      await user.click(
        screen.getByRole("button", { name: "Save and select connection" }),
      );
      await screen.findByText(/Retry resumes at/);
      expect(first.onCreated).not.toHaveBeenCalled();
      expect(
        JSON.stringify(first.client.getQueryCache().getAll()),
      ).not.toContain("private-test-sentinel");
      expect(JSON.stringify(acquired.entry)).not.toContain(
        "private-tools.example",
      );
      expect(JSON.stringify(acquired.entry)).not.toContain(
        "private-test-sentinel",
      );
      expect(storageWrite).not.toHaveBeenCalled();
      storageWrite.mockRestore();
      expect(window.location.href).not.toContain("private-test");
      first.unmount();
      store.release(acquired.entry);
      await act(async () => {
        await Promise.resolve();
      });
      const resumed = store.acquire(id, initialRunDraftState());
      expect(resumed.kind).toBe("acquired");
      if (resumed.kind !== "acquired") return;
      expect(resumed.entry.connectionCreation).toBe(progress);
      const second = dialog(api, progress);
      if (step !== "label")
        await user.type(
          screen.getByLabelText("MCP endpoint"),
          document().spec.worker!.toolsets!["mcp@1"]!.endpoint,
        );
      if (step === "credential") {
        await user.selectOptions(
          screen.getByLabelText("Authentication"),
          "new",
        );
        await user.type(
          screen.getByLabelText("New MCP credential ID"),
          "knowledge-secret",
        );
        await user.type(
          screen.getByLabelText("Header value · write only"),
          "Bearer private-test-sentinel",
        );
      }
      await user.click(
        screen.getByRole("button", { name: "Save and select connection" }),
      );
      await waitFor(() => expect(second.onCreated).toHaveBeenCalledTimes(1));
      expect(commits.size).toBe(3);
      const retries = calls.filter((call) => call.step === step);
      expect(retries).toHaveLength(2);
      expect(retries[0]!.key).toBe(retries[1]!.key);
      expect(retries[0]!.body).toEqual(retries[1]!.body);
      expect(calls.filter((call) => call.step !== step)).toHaveLength(2);
      expect(
        JSON.stringify(second.client.getMutationCache().getAll()),
      ).not.toContain("private-test-sentinel");
    },
  );
  it.each(["credential", "configuration", "label"] as const)(
    "retries an explicit rejected %s mutation",
    async (step) => {
      const { api, commits, calls } = service(step, true);
      const progress = new ToolsetConnectionCreation();
      await expect(
        progress.run(api, {
          label: "knowledge",
          document: document(),
          credential: credential(),
        }),
      ).rejects.toThrow();
      await progress.run(api, {
        label: "knowledge",
        document: document(),
        credential: credential(),
      });
      expect(commits.size).toBe(3);
      const retries = calls.filter((call) => call.step === step);
      expect(retries).toHaveLength(2);
      expect(retries[0]!.key).toBe(retries[1]!.key);
    },
  );
  it.each(["none", "existing"])(
    "creates an %s-auth connection without creating credentials",
    async (auth) => {
      const { api, calls } = service();
      const progress = new ToolsetConnectionCreation();
      const { onCreated } = dialog(api, progress);
      const user = userEvent.setup();
      expect(screen.getByLabelText("Connection name")).toHaveFocus();
      await user.type(screen.getByLabelText("Connection name"), "knowledge");
      await user.type(
        screen.getByLabelText("MCP endpoint"),
        "https://private-tools.example/mcp",
      );
      if (auth === "existing") {
        await user.selectOptions(
          screen.getByLabelText("Authentication"),
          "existing",
        );
        await user.type(
          screen.getByLabelText("Existing MCP credential (optional)"),
          "existing-secret",
        );
      }
      await user.click(screen.getByText("Advanced connection settings"));
      await user.clear(screen.getByLabelText("Request timeout (seconds)"));
      await user.type(
        screen.getByLabelText("Request timeout (seconds)"),
        "601",
      );
      await user.click(
        screen.getByRole("button", { name: "Save and select connection" }),
      );
      expect(
        await screen.findByText(/Request timeout must be an integer/),
      ).toBeVisible();
      expect(calls).toHaveLength(0);
      await user.clear(screen.getByLabelText("Request timeout (seconds)"));
      await user.type(screen.getByLabelText("Request timeout (seconds)"), "30");
      await user.click(
        screen.getByRole("button", { name: "Save and select connection" }),
      );
      await waitFor(() => expect(onCreated).toHaveBeenCalledOnce());
      expect(calls.map((call) => call.step)).toEqual([
        "configuration",
        "label",
      ]);
      expect(
        (calls[0]!.body as RuntimeConfigAuthorDocument).spec.worker!.toolsets![
          "mcp@1"
        ],
      ).toEqual({
        endpoint: "https://private-tools.example/mcp",
        connectTimeoutSeconds: 10,
        requestTimeoutSeconds: 30,
        ...(auth === "existing" ? { credential: "existing-secret" } : {}),
      });
    },
  );
  it("does not apply a completed connection after its dialog unmounts or its draft is discarded", async () => {
    const { api } = service();
    const progress = new ToolsetConnectionCreation();
    const { unmount, onCreated } = dialog(api, progress);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Connection name"), "knowledge");
    await user.type(
      screen.getByLabelText("MCP endpoint"),
      "https://private-tools.example/mcp",
    );
    unmount();
    await progress.run(api, { label: "knowledge", document: document() });
    expect(onCreated).not.toHaveBeenCalled();
    const store = new RunDraftMemoryStore("owner");
    const draft = store.acquire(
      { workflowName: "research", workflowVersion: "1" },
      initialRunDraftState(),
    );
    if (draft.kind !== "acquired") throw new Error("draft unavailable");
    store.discard(draft.entry.key);
    expect(store.keepConnectionCreation(draft.entry, progress)).toBe(false);
  });
});
