import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";

import { queryKeys } from "../../api/query-keys";
import type { WorkflowResource } from "../../api/workflows";
import { createApplicationQueryClient } from "../../app/query-client";
import { RunDraftStoreContext } from "../../run-drafts/context";
import {
  initialRunDraftState,
  RunDraftMemoryStore,
} from "../../run-drafts/memory";
import { RunDraftProvider } from "../../run-drafts/provider";
import { renderApplication } from "../../test/application";
import {
  artifactMetadata,
  artifactWrite,
  conforming,
} from "../../test/builders";
import { runStatus, workflowSummary } from "../../test/builders-execution";
import { deferred } from "../../test/deferred";
import { fakeServer, type Routes } from "../../test/fake-server";
import { failure, json, page } from "../../test/responses";
import { CSRF_TOKEN, renderWithAPI } from "../../test/render";
import { WorkflowRunForm } from "./run-form";

const digest = `sha256:${"1".repeat(64)}`;
const secondDigest = `sha256:${"2".repeat(64)}`;
const workflowDisplayName = "OpenAPI from source workspace";
const workflowDescription =
  "Build an exact OpenAPI contract from a source artifact.";

const workerConfig = {
  modelPolicy: { policyId: "worker", version: "1", digest },
  llmGateway: { gatewayId: "local", version: "1", digest },
};

const workflow: WorkflowResource = conforming("WorkflowResource", {
  ref: { name: "openapi-from-workspace", version: "fixture-1" },
  presentation: {
    displayName: workflowDisplayName,
    description: workflowDescription,
  },
  entryStage: "build",
  parameters: {
    objective: { required: true },
    audience: { required: false },
  },
  inputs: {
    source: { required: true, mediaTypes: ["application/zip"] },
  },
  outputs: {
    openapi: { required: true, mediaTypes: ["application/yaml"] },
  },
  stages: {
    build: {
      objective: "Build an OpenAPI contract from the supplied source tree.",
      instructions: { ref: "instructions/openapi-planner.md", digest },
      planner: { plannerId: "passthrough", version: "1" },
      session: "isolated",
      agents: {
        builder: {
          template: {
            templateId: "openapi-builder",
            version: "1",
            digest,
          },
          namespace: "builder",
          skills: [{ namespace: "skills", name: "openapi-analysis" }],
        },
      },
      executionConfig: { agents: { builder: workerConfig } },
      contextArtifacts: {
        source: { namespace: "inputs", name: "source", required: true },
      },
      resultArtifacts: {
        openapi: { required: true, mediaTypes: ["application/yaml"] },
      },
      workflowOutputs: { openapi: "openapi" },
      on: {
        succeeded: { kind: "succeed" },
        failed: {
          kind: "escalate",
          maxAttempts: 1,
          executionConfig: {
            ref: {
              configId: "strong-escalation",
              version: "2",
              digest,
            },
            effective: { agents: { builder: workerConfig } },
          },
          then: { kind: "fail" },
        },
        interrupted: { kind: "fail" },
      },
    },
  },
});

const workflowSelector = `${workflow.ref.name}@${workflow.ref.version}`;
const workflowRoute = `/catalog/workflows/${workflow.ref.name}/${workflow.ref.version}`;
const WORKFLOW = "GET /v1/workflows/{name}/versions/{version}";
const CREATE_RUN = "POST /v1/runs";
const nextWorkflow = conforming("WorkflowResource", {
  ref: { name: "likec4-from-workspace", version: "fixture-1" },
  entryStage: workflow.entryStage,
  parameters: workflow.parameters,
  inputs: workflow.inputs,
  outputs: workflow.outputs,
  stages: workflow.stages,
});
const sourceArtifact = artifactMetadata("projects", "source", {
  artifact: { namespace: "projects", name: "source", revision: "revision-7" },
  size: 1024,
  createdAt: "2026-08-31T12:00:00Z",
});

const pinnedRuntimeConfiguration = {
  default: {
    label: "default",
    bindingRevision: "1",
    config: { name: "contractor-empty", version: "1", digest },
  },
  labels: [],
};

const labeledRuntimeConfiguration = {
  default: pinnedRuntimeConfiguration.default,
  labels: [
    {
      label: "caido",
      bindingRevision: "4",
      config: { name: "caido", version: "1", digest: secondDigest },
    },
    {
      label: "debug",
      bindingRevision: "7",
      config: { name: "debug", version: "2", digest },
    },
  ],
};

/** The Server's answer to an accepted Run. */
function createdRun(
  runId: string,
  overrides: Partial<
    Parameters<typeof conforming<"CreateRunResponse">>[1]
  > = {},
) {
  return conforming("CreateRunResponse", {
    runId,
    state: "running",
    runtimeLabels: [],
    labels: {},
    runtimeConfiguration: pinnedRuntimeConfiguration,
    ...overrides,
  });
}

const runtimeLabelBinding = (
  label: string,
  config: { name: string; version: string; digest: string },
  revision: string,
  by: string,
) =>
  conforming("RuntimeLabelBinding", {
    label,
    config,
    revision,
    createdBy: by,
    createdAt: "2026-08-31T12:00:00Z",
    updatedBy: by,
    updatedAt: "2026-08-31T12:00:00Z",
  });

/**
 * The choices of the Run form: one published Workflow, the source ZIP, a
 * model policy, a Gateway, a credential, the default and two Runtime labels.
 */
const inventoryRoutes: Routes = {
  "GET /v1/workflows": () => json(page([workflowSummary(workflow)])),
  "GET /v1/artifacts": () => json(page([sourceArtifact])),
  "GET /v1/configurations/{kind}": ({ params }) =>
    json(
      page(
        params.kind === "model-policies"
          ? [
              conforming("ConfigurationResource", {
                ref: {
                  kind: "model-policies",
                  name: "worker-strong",
                  version: "2",
                  digest,
                },
                body: { model: "strong-model", maxTotalTokens: 100_000 },
                source: "operator",
              }),
            ]
          : params.kind === "llm-gateways"
            ? [
                conforming("ConfigurationResource", {
                  ref: {
                    kind: "llm-gateways",
                    name: "local",
                    version: "1",
                    digest,
                  },
                  body: {
                    protocol: "openai-compatible@1",
                    url: "http://192.0.2.1:4000",
                  },
                  source: "operator",
                }),
              ]
            : [],
      ),
    ),
  "GET /v1/operations/credentials": () =>
    json(
      page([
        conforming("CredentialResource", {
          credentialId: "worker-budget",
          llmGateway: { gatewayId: "local", version: "1", digest },
          label: "Worker budget",
          createdAt: "2026-08-31T12:00:00Z",
          effectivePolicy: {
            modelPolicies: [
              { policyId: "worker-strong", version: "2", digest },
            ],
            models: ["strong-model"],
          },
        }),
      ]),
    ),
  "GET /v1/operations/runtime-labels": () =>
    json(
      page([
        runtimeLabelBinding(
          "default",
          pinnedRuntimeConfiguration.default.config,
          "1",
          "system",
        ),
        runtimeLabelBinding(
          "caido",
          labeledRuntimeConfiguration.labels[0]!.config,
          "4",
          "user_local",
        ),
        runtimeLabelBinding(
          "debug",
          labeledRuntimeConfiguration.labels[1]!.config,
          "7",
          "user_local",
        ),
      ]),
    ),
};

/** The Workflow page of `workflow` with the Run form's choices and `routes`. */
function workflowRoutes(routes: Routes = {}): Routes {
  return { [WORKFLOW]: () => json(workflow), ...inventoryRoutes, ...routes };
}

/** The Run form alone, with a draft store of the signed-in owner. */
function renderRunForm(
  form: ReactElement,
  routes: Routes,
  queryClient = createApplicationQueryClient(),
) {
  return renderWithAPI(
    <RunDraftProvider ownerId="user_local">{form}</RunDraftProvider>,
    { server: fakeServer(routes), queryClient },
  );
}

/** A response the client never receives: the Run may exist. */
function responseLost(): never {
  throw new TypeError("response lost");
}

async function openRunSetup() {
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "Configure Run" }));
}

describe("Workflow overview and Run drawer", () => {
  const first = { ...workflow, ref: { ...workflow.ref, version: "1" } };
  const second = { ...workflow, ref: { ...workflow.ref, version: "2" } };
  const route = `/catalog/workflows/${workflow.ref.name}`;
  function overviewRoutes(versionsUnavailable = false): Routes {
    return {
      ...inventoryRoutes,
      "GET /v1/workflows": ({ url }) =>
        versionsUnavailable
          ? failure(503, "unavailable", "Unavailable")
          : url.searchParams.has("cursor")
            ? json(page([workflowSummary(second)]))
            : json(page([workflowSummary(first)], "older-page")),
      [WORKFLOW]: ({ params }) => json(params.version === "2" ? second : first),
    };
  }

  it("shows the contract before setup, loads all versions and preserves a pinned older route", async () => {
    const { router, sent } = renderApplication(`${route}/1`, {
      server: overviewRoutes(),
    });

    expect(
      await screen.findByRole("heading", { name: "Inputs and results" }),
    ).toBeVisible();
    expect(screen.getByRole("region", { name: "Input files" })).toBeVisible();
    expect(
      screen.getByRole("region", { name: "Declared outputs" }),
    ).toBeVisible();
    expect(
      screen.queryByText("No authored description."),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Start Workflow Run" }),
    ).not.toBeInTheDocument();
    expect(sent("GET", "/v1/artifacts")).toHaveLength(0);
    expect(await screen.findByText("Earlier version")).toBeVisible();
    expect(
      screen.getByRole("combobox", { name: `Version of ${workflow.ref.name}` }),
    ).toHaveValue("1");
    expect(router.state.location.pathname).toBe(`${route}/1`);
    expect(sent("GET", "/v1/workflows")).toHaveLength(2);
    const user = userEvent.setup();
    await user.selectOptions(
      screen.getByRole("combobox", { name: `Version of ${workflow.ref.name}` }),
      "2",
    );
    expect(await screen.findByText("Latest")).toBeVisible();
    expect(router.state.location.pathname).toBe(`${route}/2`);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("retains optional and required values across closing, nested dialogs and exact version changes", async () => {
    renderApplication(`${route}/1`, { server: overviewRoutes() });

    const user = userEvent.setup();
    const trigger = await screen.findByRole("button", {
      name: "Configure Run",
    });
    await user.click(trigger);
    let drawer = screen.getByRole("dialog", { name: "Configure Run" });
    expect(
      within(drawer).getByRole("button", { name: "Close Run setup" }),
    ).toHaveFocus();
    // The exact version stays visible, and copyable, before the Run starts.
    expect(
      drawer.querySelector(".workflow-drawer-heading code"),
    ).toHaveTextContent(/^openapi-from-workspace@1$/);
    expect(
      within(drawer).getByRole("button", { name: "Copy workflow version" }),
    ).toBeVisible();
    const objective = within(drawer).getByLabelText(/^objective/);
    const source = within(drawer).getByLabelText(/^source/);
    expect(
      source.compareDocumentPosition(objective) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    await user.type(objective, "Retained version one");
    await user.click(
      within(drawer).getByText("Optional parameters", { exact: true }),
    );
    await user.click(
      within(drawer).getByLabelText(/Include optional audience/),
    );
    await user.type(
      within(drawer).getByLabelText("audience"),
      "Security reviewers",
    );
    await user.click(
      within(drawer).getByRole("button", {
        name: "Upload local file for source",
      }),
    );
    expect(
      screen.getByRole("dialog", { name: "Upload local file for source" }),
    ).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog", { name: "Configure Run" })).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(trigger).toHaveFocus());
    await screen.findByText("Earlier version");
    await user.selectOptions(
      screen.getByRole("combobox", { name: `Version of ${workflow.ref.name}` }),
      "2",
    );
    await screen.findByText("Latest");
    await openRunSetup();
    expect(screen.getByLabelText(/^objective/)).toHaveValue("");
    await user.keyboard("{Escape}");
    await user.selectOptions(
      screen.getByRole("combobox", { name: `Version of ${workflow.ref.name}` }),
      "1",
    );
    await screen.findByText("Earlier version");
    await openRunSetup();
    drawer = screen.getByRole("dialog", { name: "Configure Run" });
    expect(within(drawer).getByLabelText(/^objective/)).toHaveValue(
      "Retained version one",
    );
    expect(within(drawer).getByLabelText("audience")).toBeVisible();
    expect(within(drawer).getByLabelText("audience")).toHaveValue(
      "Security reviewers",
    );
  });

  it("keeps exact setup available if the version inventory fails without claiming Latest", async () => {
    renderApplication(`${route}/1`, { server: overviewRoutes(true) });

    await screen.findByRole("button", { name: "Retry versions" });
    expect(
      screen.getByRole("combobox", { name: `Version of ${workflow.ref.name}` }),
    ).toHaveValue("1");
    expect(screen.queryByText("Latest")).not.toBeInTheDocument();
    await openRunSetup();
    expect(screen.getByRole("dialog", { name: "Configure Run" })).toBeVisible();
  });

  it("protects pending submission and preserves the exact retry identity after reopening", async () => {
    const noInputs = { ...first, parameters: {}, inputs: {} };
    let rejectSubmission: (error: Error) => void = () => {};
    const keys: (string | null)[] = [];
    renderApplication(`${route}/1#workflow-run-setup`, {
      server: {
        ...inventoryRoutes,
        [WORKFLOW]: () => json(noInputs),
        [CREATE_RUN]: ({ request }) => {
          keys.push(request.headers.get("Idempotency-Key"));
          if (keys.length === 1)
            return new Promise<Response>((_, reject) => {
              rejectSubmission = reject;
            });
          return responseLost();
        },
      },
    });

    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "Start Workflow Run" }),
    );
    await waitFor(() => expect(keys).toHaveLength(1));
    expect(
      screen.getByRole("button", { name: "Close Run setup" }),
    ).toBeDisabled();
    await user.keyboard("{Escape}");
    expect(screen.getByRole("dialog", { name: "Configure Run" })).toBeVisible();
    rejectSubmission(new TypeError("response lost"));
    await screen.findByRole("button", { name: "Retry same request" });
    await user.keyboard("{Escape}");
    await openRunSetup();
    await user.click(
      screen.getByRole("button", { name: "Retry same request" }),
    );
    await waitFor(() => expect(keys).toHaveLength(2));
    expect(keys[0]).toMatch(/^run-ui-/);
    expect(keys[1]).toBe(keys[0]);
  });
});

describe("Workflow routes", () => {
  it.each([workflowDisplayName, "View workflow"])(
    "preserves Workflow search and return state through the %s link",
    async (linkName) => {
      const path = "/catalog/workflows?q=openapi&keep=yes";
      const { router } = renderApplication(path, { server: workflowRoutes() });

      const returnState = { from: "catalog-navigation" };
      await act(async () => {
        await router.navigate(path, { replace: true, state: returnState });
      });
      const user = userEvent.setup();
      await user.click(await screen.findByRole("link", { name: linkName }));
      await user.click(
        await screen.findByRole("link", { name: /Workflow search/ }),
      );
      expect(router.state.location.pathname).toBe("/catalog/workflows");
      expect(router.state.location.search).toBe("?q=openapi&keep=yes");
      expect(router.state.location.state).toEqual(returnState);
      expect(await screen.findByLabelText("Search workflows")).toHaveValue(
        "openapi",
      );
    },
  );

  it.each([
    { projectId: undefined, resolveBeforeReplacement: false },
    { projectId: "project_example", resolveBeforeReplacement: false },
    { projectId: undefined, resolveBeforeReplacement: true },
    { projectId: "project_example", resolveBeforeReplacement: true },
  ])(
    "preserves the replacement form after an earlier success: %j",
    async ({ projectId, resolveBeforeReplacement }) => {
      let resolveSubmission!: (response: Response) => void;
      const pending = new Promise<Response>((resolve) => {
        resolveSubmission = resolve;
      });
      const submissions: Request[] = [];
      const submit = ({ request }: { request: Request }) => {
        submissions.push(request);
        if (submissions.length === 1) return pending;
        throw new TypeError("replacement response lost");
      };
      const queryClient = createApplicationQueryClient();
      const invalidation = deferred();
      const invalidate = vi.spyOn(queryClient, "invalidateQueries");
      if (resolveBeforeReplacement)
        invalidate.mockReturnValue(invalidation.promise);
      const store = new RunDraftMemoryStore("user_local");
      const formPath =
        projectId === undefined
          ? `${workflowRoute}/run`
          : `/projects/${projectId}/workflows/${workflow.ref.name}/${workflow.ref.version}/run`;
      const { router, user } = renderWithAPI(null, {
        path: formPath,
        queryClient,
        server: fakeServer({
          "GET /v1/artifacts": () => json(page([sourceArtifact])),
          "GET /v1/projects/{projectId}/artifacts": () =>
            json(page([sourceArtifact])),
          [CREATE_RUN]: submit,
          "POST /v1/projects/{projectId}/runs": submit,
        }),
        routes: [
          {
            path: formPath,
            element: (
              <RunDraftStoreContext.Provider value={store}>
                <WorkflowRunForm
                  workflow={workflow}
                  {...(projectId === undefined ? {} : { projectId })}
                />
              </RunDraftStoreContext.Provider>
            ),
          },
          { path: "/away", element: <p>Away</p> },
          { path: "/runs/:runId", element: <p>Created Run</p> },
        ],
      });

      async function edit(objective: string) {
        await user.type(await screen.findByLabelText(/^objective/i), objective);
        await screen.findByRole("option", {
          name: /projects\/source@revision-7/,
        });
        await user.selectOptions(
          screen.getByLabelText(/^source/i),
          "projects/source@revision-7",
        );
      }
      await edit("Original request");
      await user.click(
        screen.getByRole("button", { name: /^Start (Project )?Workflow Run$/ }),
      );
      await waitFor(() => expect(submissions).toHaveLength(1));
      const success = () =>
        resolveSubmission(
          json(
            createdRun("run_old", projectId === undefined ? {} : { projectId }),
            202,
          ),
        );
      if (resolveBeforeReplacement) {
        await act(async () => {
          success();
        });
        await waitFor(() => expect(invalidate).toHaveBeenCalled());
      }
      await act(async () => {
        await router.navigate("/away");
      });
      await act(async () => {
        await router.navigate(formPath);
      });
      await user.click(
        await screen.findByRole("button", { name: "Discard saved draft" }),
      );
      await user.click(
        screen.getByRole("button", { name: "Discard Run draft" }),
      );
      await edit("Replacement request");
      await user.click(
        screen.getByRole("button", { name: /^Start (Project )?Workflow Run$/ }),
      );
      await screen.findByRole("button", { name: "Retry same request" });
      const acquisition = store.acquire(
        {
          workflowName: workflow.ref.name,
          workflowVersion: workflow.ref.version,
          ...(projectId === undefined ? {} : { projectId }),
        },
        initialRunDraftState(),
      );
      if (acquisition.kind !== "acquired")
        throw new Error("missing replacement");
      const replacement = acquisition.entry;
      const replacementState = structuredClone(replacement.state);
      await act(async () => {
        if (!resolveBeforeReplacement) success();
        invalidation.resolve();
      });
      await waitFor(() =>
        expect(invalidate).toHaveBeenCalledTimes(
          projectId === undefined ? 1 : 3,
        ),
      );
      expect(router.state.location.pathname).toBe(formPath);
      expect(store.isCurrent(replacement)).toBe(true);
      expect(replacement.state).toEqual(replacementState);
      expect(replacement.ambiguousSubmission).toBe(true);
      expect(screen.getByLabelText(/^objective/i)).toHaveValue(
        "Replacement request",
      );
      expect(screen.getByLabelText(/^source/i)).toHaveValue(
        "projects/source@revision-7",
      );
      await user.click(
        screen.getByRole("button", { name: "Retry same request" }),
      );
      await waitFor(() => expect(submissions).toHaveLength(3));
      expect(submissions[2]!.headers.get("Idempotency-Key")).toBe(
        submissions[1]!.headers.get("Idempotency-Key"),
      );
      expect(submissions[1]!.headers.get("Idempotency-Key")).not.toBe(
        submissions[0]!.headers.get("Idempotency-Key"),
      );
      expect(
        submissions.every(
          (request) =>
            new URL(request.url).pathname ===
            (projectId === undefined
              ? "/v1/runs"
              : `/v1/projects/${projectId}/runs`),
        ),
      ).toBe(true);
    },
  );

  it("requires slot-by-slot review when one MIME candidate is suggested twice", async () => {
    const reportArtifact = artifactMetadata("reports", "review", {
      ...sourceArtifact,
      artifact: {
        namespace: "reports",
        name: "review",
        revision: "revision-report-1",
      },
      mediaType: "text/markdown",
    });
    const exactRef = "reports/review@revision-report-1";
    const reviewWorkflow: WorkflowResource = {
      ...workflow,
      parameters: {},
      inputs: {
        dependency_report: {
          required: true,
          mediaTypes: ["text/markdown"],
        },
        project_report: {
          required: true,
          mediaTypes: ["text/markdown"],
        },
      },
    };
    const { user } = renderRunForm(
      <WorkflowRunForm
        workflow={reviewWorkflow}
        initialArtifactSelections={{
          dependency_report: exactRef,
          project_report: exactRef,
        }}
        initialArtifacts={[reportArtifact]}
      />,
      { "GET /v1/artifacts": () => json(page([reportArtifact])) },
    );

    expect(await screen.findByText("Review 2")).toBeVisible();
    expect(
      screen.getAllByText(/Suggested only because text\/markdown/),
    ).toHaveLength(2);
    expect(
      screen.getAllByText(/A format match does not check what the file/),
    ).toHaveLength(2);
    expect(
      screen.getAllByRole("link", { name: "Preview file details" }),
    ).toHaveLength(2);

    await user.click(
      screen.getByRole("button", {
        name: "Confirm input for dependency_report",
      }),
    );
    expect(screen.getByText("Review 1")).toBeVisible();
    await user.click(
      screen.getByRole("button", {
        name: "Confirm input for project_report",
      }),
    );
    expect(screen.getByText("Ready", { exact: true })).toBeVisible();
  });

  it.each([undefined, "project_example"])(
    "only offers compatible input revisions in scope %s",
    async (projectId) => {
      const textArtifact = {
        ...sourceArtifact,
        artifact: { ...sourceArtifact.artifact, name: "readme" },
        mediaType: "text/plain",
      };
      const list = () => json(page([sourceArtifact, textArtifact]));
      const { user } = renderRunForm(
        <WorkflowRunForm
          workflow={{
            ...workflow,
            inputs: {
              ...workflow.inputs,
              document: {
                required: false,
                mediaTypes: ["application/pdf"],
              },
              any: { required: false, mediaTypes: ["*/*"] },
            },
          }}
          {...(projectId === undefined ? {} : { projectId })}
        />,
        {
          "GET /v1/artifacts": list,
          "GET /v1/projects/{projectId}/artifacts": list,
        },
      );

      const source = screen.getByRole("combobox", { name: "source required" });
      await within(source).findByRole("option", { name: /source@revision-7/ });
      expect(within(source).getAllByRole("option")).toHaveLength(2);
      expect(
        within(source).queryByRole("option", { name: /readme/ }),
      ).toBeNull();
      const document = screen.getByRole("combobox", { name: "document" });
      expect(within(document).getAllByRole("option")).toHaveLength(1);
      expect(
        screen.getByText(
          projectId === undefined
            ? /No matching files found/
            : /No matching materials found/,
        ),
      ).toBeVisible();
      const any = screen.getByRole("combobox", { name: "any" });
      expect(within(any).getAllByRole("option")).toHaveLength(3);
      expect(screen.queryByRole("option", { name: /Incompatible/ })).toBeNull();
      const git = screen.getByRole("button", { name: "Import Git for source" });
      expect(git.querySelector("svg")).not.toBeNull();
      expect(
        screen.queryByRole("button", { name: "Import Git for document" }),
      ).toBeNull();
      await user.click(git);
      expect(
        screen.getByRole("dialog", { name: "Import Git repository" }),
      ).toBeVisible();
    },
  );

  it("isolates infinite Run inventories from finite Operations picker cache", async () => {
    const queryClient = createApplicationQueryClient();
    queryClient.setQueryData(queryKeys.configurations.picker("llm-gateways"), {
      items: [],
      page: { hasMore: false },
    });
    const { requests, user } = renderRunForm(
      <WorkflowRunForm workflow={workflow} />,
      inventoryRoutes,
      queryClient,
    );
    const paths = () =>
      requests.map((request) => new URL(request.url).pathname);

    expect(
      await screen.findByRole("heading", { name: "Start Workflow Run" }),
    ).toBeInTheDocument();
    expect(await screen.findByText("0/2")).toBeInTheDocument();
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    expect(paths()).toEqual(["/v1/artifacts"]);

    await user.click(screen.getByText("Execution overrides", { exact: true }));
    expect(
      await screen.findAllByRole("option", { name: /worker-strong@2/ }),
    ).toHaveLength(2);
    expect(paths()).not.toContain("/v1/operations/runtime-labels");
  });

  it("loads all published Workflow pages before presenting grouped cards", async () => {
    const { router } = renderApplication("/catalog/workflows", {
      server: {
        "GET /v1/workflows": ({ url }) =>
          url.searchParams.get("q") === "likec4" ||
          url.searchParams.get("cursor") === "next-workflow"
            ? json(page([workflowSummary(nextWorkflow)]))
            : json(page([workflowSummary(workflow)], "next-workflow")),
      },
    });

    expect(
      await screen.findByRole("link", {
        name: workflowDisplayName,
      }),
    ).toHaveAttribute("href", workflowRoute);
    expect(screen.getByText(workflowDescription)).toBeVisible();
    expect(screen.getByText(workflowSelector)).toBeInTheDocument();
    const user = userEvent.setup();
    expect(
      await screen.findByRole("link", { name: nextWorkflow.ref.name }),
    ).toBeInTheDocument();
    expect(screen.queryByText("No authored description.")).toBeNull();
    expect(
      screen.getByText("2 workflows · 2 published versions"),
    ).toBeVisible();

    await user.type(screen.getByLabelText("Search workflows"), "likec4");
    await waitFor(() =>
      expect(router.state.location.search).toContain("q=likec4"),
    );
    expect(router.state.location.search).not.toContain("cursor=");
    expect(router.state.location.search).not.toContain("page=");
    expect(
      await screen.findByRole("link", { name: nextWorkflow.ref.name }),
    ).toBeVisible();
  });

  it("submits declared strings, an exact Artifact, and published overrides", async () => {
    const posts: Array<{ request: Request; body: unknown }> = [];
    const labels = {
      "eval.id": "eval-ui-01",
      "eval.leg": "a",
      "eval.name": "openapi-browser",
      purpose: "eval",
    };
    const { router } = renderApplication(workflowRoute, {
      server: workflowRoutes({
        [CREATE_RUN]: async ({ request }) => {
          posts.push({ request, body: await request.clone().json() });
          return json(
            createdRun("run_openapi", {
              runtimeLabels: ["caido", "debug"],
              labels,
              runtimeConfiguration: labeledRuntimeConfiguration,
            }),
            202,
          );
        },
        "GET /v1/runs/{runId}": () =>
          json(
            runStatus("run_openapi", {
              workflow: workflowSelector,
              state: "running",
              deletable: false,
              runtimeLabels: ["caido", "debug"],
              labels,
              runtimeConfiguration: labeledRuntimeConfiguration,
            }),
          ),
        "GET /v1/runs/{runId}/artifacts": () => json(page([])),
      }),
    });

    expect(
      await screen.findByRole("heading", {
        name: workflowDisplayName,
      }),
    ).toBeInTheDocument();
    await screen.findByText("Scheduler transitions");
    expect(
      screen.getByText(
        (_, element) =>
          element?.tagName === "SPAN" &&
          element.textContent?.includes("escalate up to 1 attempt using") ===
            true,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("strong-escalation@2", { selector: "code" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /escalate/i }),
    ).not.toBeInTheDocument();

    await openRunSetup();
    const user = userEvent.setup();
    await user.click(screen.getByText("Runtime placement", { exact: true }));
    expect(
      await screen.findByText("Default · always applied"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: /debug/ }));
    await user.click(screen.getByRole("checkbox", { name: /caido/ }));
    await user.click(screen.getByText("Run metadata", { exact: true }));
    await user.click(
      screen.getByRole("button", { name: "Add eval metadata preset" }),
    );
    expect(screen.getByLabelText("Run metadata label key 1")).toHaveValue(
      "purpose",
    );
    await user.type(
      screen.getByLabelText("Run metadata label value 2"),
      "openapi-browser",
    );
    await user.type(
      screen.getByLabelText("Run metadata label value 3"),
      "eval-ui-01",
    );
    await user.type(screen.getByLabelText("Run metadata label value 4"), "a");
    await user.type(
      screen.getByLabelText(/^objective/i),
      "Build public API\nInclude exposed routes.",
    );
    await screen.findByRole("option", {
      name: /projects\/source@revision-7/,
    });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(screen.getByText("Execution overrides", { exact: true }));
    const workerOverrides = screen.getByRole("group", { name: "Workers" });
    await user.selectOptions(
      within(workerOverrides).getByLabelText("Model policy"),
      "worker-strong@2",
    );
    await user.selectOptions(
      within(workerOverrides).getByLabelText("LLM Gateway"),
      "local@1",
    );
    await user.selectOptions(
      within(workerOverrides).getByLabelText("Credential"),
      "worker-budget",
    );
    expect(screen.queryByLabelText(/Gateway URL/i)).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );

    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/runs/run_openapi"),
    );
    expect(posts).toHaveLength(1);
    expect(posts[0]?.request.headers.get("X-CSRF-Token")).toBe(CSRF_TOKEN);
    expect(posts[0]?.request.headers.get("Idempotency-Key")).toMatch(
      /^run-ui-[0-9a-f]{32}$/,
    );
    expect(posts[0]?.body).toEqual({
      workflow: workflowSelector,
      runtimeLabels: ["caido", "debug"],
      labels: {
        "eval.id": "eval-ui-01",
        "eval.leg": "a",
        "eval.name": "openapi-browser",
        purpose: "eval",
      },
      parameters: { objective: "Build public API\nInclude exposed routes." },
      artifacts: {
        source: {
          namespace: "projects",
          name: "source",
          revision: "revision-7",
        },
      },
      executionConfig: {
        workers: {
          modelPolicy: "worker-strong@2",
          llmGateway: "local@1",
          credential: "worker-budget",
        },
      },
    });
    expect(
      await screen.findByRole("button", { name: "Copy Run ID" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Workflow Run", { exact: true }),
    ).toBeInTheDocument();
    expect(await screen.findByText("binding revision 7")).toBeInTheDocument();
    const metadata = screen
      .getByRole("heading", { name: "Run metadata labels" })
      .closest("section");
    expect(metadata).not.toBeNull();
    expect(metadata).toHaveTextContent("eval.id:eval-ui-01");
    expect(metadata).toHaveTextContent("eval.leg:a");
    expect(metadata).toHaveTextContent("Labels are fixed at creation");
  });

  it("blocks a mutation while required declared fields are missing", async () => {
    let postCount = 0;
    renderApplication(workflowRoute, {
      server: workflowRoutes({
        "GET /v1/configurations/{kind}": () => json(page([])),
        "GET /v1/operations/credentials": () => json(page([])),
        [CREATE_RUN]: () => {
          postCount += 1;
          return responseLost();
        },
      }),
    });

    await userEvent
      .setup()
      .click(
        await screen.findByText("Agents, execution settings and transitions"),
      );
    expect(
      await screen.findByRole("link", { name: "skills/openapi-analysis" }),
    ).toHaveAttribute("href", "/artifacts/skills/openapi-analysis");
    await openRunSetup();
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    expect(
      await screen.findByText("Required string parameter is missing."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Choose a revision for this required input."),
    ).toBeInTheDocument();
    expect(postCount).toBe(0);
    await user.click(screen.getByText("Execution overrides", { exact: true }));
    expect(
      await screen.findByText("No ModelPolicy versions published."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No LLMGatewayConfig versions published."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No active credentials available."),
    ).toBeInTheDocument();
  });

  it("reuses one key for exact response-loss retry and rotates it after edits", async () => {
    const keys: string[] = [];
    const { router } = renderApplication(workflowRoute, {
      server: workflowRoutes({
        [CREATE_RUN]: ({ request }) => {
          keys.push(request.headers.get("Idempotency-Key") ?? "missing");
          if (keys.length < 3) return responseLost();
          return json(
            createdRun("run_changed", { labels: { "eval.leg": "b" } }),
            202,
          );
        },
      }),
    });

    await openRunSetup();
    const user = userEvent.setup();
    const objective = await screen.findByLabelText(/^objective/i);
    await user.type(objective, "Initial objective");
    await user.click(screen.getByText("Run metadata", { exact: true }));
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 1"),
      "eval.leg",
    );
    const labelValue = screen.getByLabelText("Run metadata label value 1");
    await user.type(labelValue, "a");
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await screen.findByRole("button", { name: "Retry same request" });
    await user.type(screen.getByLabelText("Add a label"), "pending:restored");
    await router.navigate("/artifacts");
    await screen.findByRole("heading", { name: "Files" });
    await router.navigate(workflowRoute);
    await openRunSetup();
    expect(await screen.findByLabelText(/^objective/i)).toHaveValue(
      "Initial objective",
    );
    expect(screen.getByLabelText("Run metadata label value 1")).toHaveValue(
      "a",
    );
    expect(screen.getByLabelText("Add a label")).toHaveValue(
      "pending:restored",
    );
    expect(screen.getByLabelText(/^source/i)).toHaveValue(
      "projects/source@revision-7",
    );
    await user.click(
      await screen.findByRole("button", { name: "Retry same request" }),
    );
    expect(
      await screen.findByRole("button", { name: "Retry same request" }),
    ).toBeInTheDocument();
    expect(keys[0]).toBe(keys[1]);

    const restoredLabelValue = screen.getByLabelText(
      "Run metadata label value 1",
    );
    await user.clear(restoredLabelValue);
    await user.type(restoredLabelValue, "b");
    await user.click(
      screen.getByRole("button", {
        name: "Start changed draft with a new key",
      }),
    );
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/runs/run_changed"),
    );
    expect(keys).toHaveLength(3);
    expect(keys[2]).not.toBe(keys[1]);
  });

  it("authors editable colon badges and key-only labels without losing pending input on submit", async () => {
    const requests: unknown[] = [];
    renderApplication(workflowRoute, {
      server: workflowRoutes({
        [CREATE_RUN]: async ({ request }) => {
          requests.push(await request.json());
          return responseLost();
        },
      }),
    });

    await openRunSetup();
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Badge labels",
    );
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(screen.getByText("Run metadata", { exact: true }));
    const composer = screen.getByLabelText("Add a label");
    await user.type(
      composer,
      "team:platform{Enter}debug{Enter}endpoint:https://host:8443/a=b{Enter}",
    );
    expect(requests).toHaveLength(0);
    expect(screen.getByLabelText("Run metadata label key 1")).toHaveValue(
      "team",
    );
    expect(screen.getByLabelText("Run metadata label value 2")).toHaveValue("");
    expect(screen.getByLabelText("Run metadata label value 3")).toHaveValue(
      "https://host:8443/a=b",
    );
    await user.clear(screen.getByLabelText("Run metadata label value 1"));
    await user.type(
      screen.getByLabelText("Run metadata label value 1"),
      "infra{Enter}",
    );
    expect(requests).toHaveLength(0);
    await user.click(
      screen.getByRole("button", { name: "Remove Run metadata label 3" }),
    );
    await user.type(composer, "temporary:");
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    expect(screen.getAllByLabelText(/^Run metadata label key /)).toHaveLength(
      3,
    );
    expect(screen.getByLabelText("Run metadata label value 3")).toHaveValue("");
    await user.click(
      screen.getByRole("button", { name: "Remove Run metadata label 3" }),
    );
    await user.type(composer, "release:next");
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );
    await screen.findByRole("button", { name: "Retry same request" });
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      labels: { team: "infra", debug: "", release: "next" },
    });
  });

  it("uploads a local UserScope file into its originating exact input slot", async () => {
    let stored = false;
    const uploadedArtifact = artifactMetadata("inputs", "service-source", {
      artifact: {
        namespace: "inputs",
        name: "service-source",
        revision: "revision-upload-1",
      },
      size: 3,
      createdAt: "2026-09-06T20:00:00Z",
    });
    const { sent } = renderApplication(workflowRoute, {
      server: workflowRoutes({
        "GET /v1/artifacts": () => json(page(stored ? [uploadedArtifact] : [])),
        "PUT /v1/artifacts/{namespace}/{name}": async ({ request, params }) => {
          expect(params).toEqual({
            namespace: "inputs",
            name: "service-source",
          });
          expect(Buffer.from(await request.arrayBuffer()).toString()).toBe(
            "zip",
          );
          expect(request.headers.get("If-None-Match")).toBe("*");
          expect(request.headers.get("Content-Type")).toBe("application/zip");
          stored = true;
          return json(artifactWrite(uploadedArtifact), 201, {
            ETag: '"revision-upload-1"',
          });
        },
      }),
      strict: true,
    });
    const puts = () => sent("PUT", "/v1/artifacts/inputs/service-source");

    await openRunSetup();
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Keep this field while uploading",
    );
    await user.click(
      screen.getByRole("button", { name: "Upload local file for source" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Upload local file for source",
    });
    // The dialog title labels the form; no "Artifact" heading of its own.
    expect(
      within(dialog).getByRole("form", {
        name: "Upload local file for source",
      }),
    ).toBeVisible();
    expect(within(dialog).queryByText(/artifact/i)).not.toBeInTheDocument();
    await user.upload(
      within(dialog).getByLabelText("Drop a file here"),
      new File(["zip"], "service-source.zip", {
        type: "application/zip",
      }),
    );
    expect(within(dialog).getByLabelText("Namespace")).toHaveValue("inputs");
    expect(within(dialog).getByLabelText("Namespace")).toBeDisabled();
    expect(within(dialog).getByLabelText("Name")).toHaveValue("service-source");
    await user.click(
      within(dialog).getByRole("button", {
        name: "Upload and select",
      }),
    );

    await waitFor(() => expect(dialog).not.toBeInTheDocument());
    expect(puts()).toHaveLength(1);
    expect(screen.getByLabelText(/^objective/i)).toHaveValue(
      "Keep this field while uploading",
    );
    expect(screen.getByLabelText(/^source/i)).toHaveValue(
      "inputs/service-source@revision-upload-1",
    );

    await user.click(
      screen.getByRole("button", { name: "Discard saved draft" }),
    );
    const confirmation = screen.getByRole("alertdialog", {
      name: "Discard this Run draft?",
    });
    expect(
      within(confirmation).getByRole("button", { name: "Keep editing" }),
    ).toHaveFocus();
    await user.click(
      within(confirmation).getByRole("button", {
        name: "Discard Run draft",
      }),
    );
    expect(await screen.findByLabelText(/^objective/i)).toHaveValue("");
    expect(screen.getByLabelText(/^source/i)).toHaveValue("");
  });

  it("aborts a pending local upload without binding it into the retained draft", async () => {
    const started = deferred();
    let uploadAborted = false;
    renderApplication(workflowRoute, {
      server: workflowRoutes({
        "GET /v1/artifacts": () => json(page([])),
        "PUT /v1/artifacts/{namespace}/{name}": async ({ request, params }) => {
          expect(params.name).toBe("cancel-source");
          started.resolve();
          return await new Promise<Response>((_resolve, reject) => {
            const abort = () => {
              uploadAborted = true;
              reject(new DOMException("Aborted", "AbortError"));
            };
            if (request.signal.aborted) {
              abort();
              return;
            }
            request.signal.addEventListener("abort", abort, { once: true });
          });
        },
      }),
      strict: true,
    });

    await openRunSetup();
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Retain this draft after cancellation",
    );
    await user.click(
      screen.getByRole("button", { name: "Upload local file for source" }),
    );
    const dialog = screen.getByRole("dialog", {
      name: "Upload local file for source",
    });
    await user.upload(
      within(dialog).getByLabelText("Drop a file here"),
      new File(["zip"], "cancel-source.zip", { type: "application/zip" }),
    );
    await user.click(
      within(dialog).getByRole("button", {
        name: "Upload and select",
      }),
    );
    await started.promise;
    await user.click(
      within(dialog).getByRole("button", {
        name: "Close local file upload",
      }),
    );

    await waitFor(() => expect(uploadAborted).toBe(true));
    expect(dialog).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^objective/i)).toHaveValue(
      "Retain this draft after cancellation",
    );
    expect(screen.getByLabelText(/^source/i)).toHaveValue("");
  });

  it("blocks duplicate and reserved Run metadata labels before mutation", async () => {
    let postCount = 0;
    renderApplication(workflowRoute, {
      server: workflowRoutes({
        [CREATE_RUN]: () => {
          postCount += 1;
          return responseLost();
        },
      }),
    });

    await openRunSetup();
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText(/^objective/i),
      "Validate labels",
    );
    await screen.findByRole("option", { name: /projects\/source@revision-7/ });
    await user.selectOptions(
      screen.getByLabelText(/^source/i),
      "projects/source@revision-7",
    );
    await user.click(screen.getByText("Run metadata", { exact: true }));
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 1"),
      "contractor.secret",
    );
    await user.type(
      screen.getByLabelText("Run metadata label value 1"),
      "must-not-submit",
    );
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 2"),
      "eval.id",
    );
    await user.type(screen.getByLabelText("Run metadata label value 2"), "one");
    await user.click(
      screen.getByRole("button", { name: "Add metadata label" }),
    );
    await user.type(
      screen.getByLabelText("Run metadata label key 3"),
      "eval.id",
    );
    await user.type(screen.getByLabelText("Run metadata label value 3"), "two");
    await user.click(
      screen.getByRole("button", { name: "Start Workflow Run" }),
    );

    expect(
      await screen.findByText("The contractor. prefix is reserved."),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Label key eval.id is duplicated."),
    ).toHaveLength(2);
    expect(postCount).toBe(0);
  });
});
