import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { components } from "../../api/generated/public";
import { renderApplication } from "../../test/application";
import { conforming } from "../../test/builders";
import { workflowResource } from "../../test/builders-execution";
import { fakeServer } from "../../test/fake-server";
import { emptyPageOrNotFound, failure, json, page } from "../../test/responses";

type Schemas = components["schemas"];

const digest = `sha256:${"1".repeat(64)}`;
const secondDigest = `sha256:${"2".repeat(64)}`;
const description = "Researches the supplied evidence";
const prompt =
  "# Careful researcher\n\nRead the evidence first.\n\n![remote](https://example.test/image.png)\n<script>alert('no')</script>";

/** The researcher Agent template at `version`; `tool` makes it a tool agent. */
function agentTemplate(
  version = "1",
  tool = false,
): Schemas["ConfigurationResource"] {
  const exactDigest = version === "2" ? secondDigest : digest;
  const shared = {
    description,
    toolsets: [{ ref: "run-artifacts@1", tools: ["read_artifact"] }],
    skills: [{ namespace: "skills" as const, name: "research" }],
    sandboxProfile: "local-workdir@1",
  };
  const body: Schemas["AgentTemplateBody"] = tool
    ? {
        ...shared,
        runtime: "tool@1",
        execution: {
          tool: "read_artifact",
          arguments: {},
          resultArtifact: "report",
          timeoutSeconds: 60,
        },
      }
    : {
        ...shared,
        runtime: "adk@1",
        instructions: {
          ref: "instructions/researcher.md",
          digest: exactDigest,
        },
      };
  return conforming("ConfigurationResource", {
    ref: {
      kind: "agent-templates",
      name: "researcher",
      version,
      digest: exactDigest,
    },
    source: "operator",
    body,
  });
}

/** Resolves once `signal` aborts. */
function aborted(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) resolve();
    else signal.addEventListener("abort", () => resolve(), { once: true });
  });
}

let currentRouter: ReturnType<typeof renderApplication>["router"];

function setup(
  path: string,
  options: {
    mismatch?: boolean;
    failAgentList?: boolean;
    slowSearch?: string;
    paginated?: boolean;
    paginatedUsage?: boolean;
    toolAgent?: boolean;
  } = {},
) {
  const abortedSearches: string[] = [];
  const server = fakeServer(
    {
      "GET /v1/configurations/agent-templates/{name}/versions/{version}/workflow-bindings":
        ({ url, params }) =>
          json(
            page(
              [
                params.version === "2"
                  ? {
                      workflow: { name: "likec4-from-source", version: "beta" },
                      stage: "model",
                      logicalWorker: "architect",
                    }
                  : {
                      workflow: { name: "openapi-from-source", version: "1" },
                      stage: "analyze",
                      logicalWorker: "builder",
                    },
              ],
              options.paginatedUsage && !url.searchParams.has("cursor")
                ? "usage-next"
                : undefined,
            ),
          ),
      "GET /v1/configurations/agent-templates/{name}/versions/{version}/instructions":
        ({ params }) => {
          const exactDigest = params.version === "2" ? secondDigest : digest;
          return json({
            template: {
              templateId: "researcher",
              version: params.version,
              digest: options.mismatch ? secondDigest : exactDigest,
            },
            instructions: {
              ref: "instructions/researcher.md",
              digest: exactDigest,
              text: params.version === "2" ? "# Second version" : prompt,
            },
          });
        },
      "GET /v1/configurations/{kind}/{name}/versions/{version}": ({ params }) =>
        json(agentTemplate(params.version, options.toolAgent)),
      "GET /v1/configurations/{kind}": async ({ request, url, params }) => {
        if (params.kind !== "agent-templates") return json(page([]));
        const exactName = url.searchParams.get("name");
        const cursor = url.searchParams.get("cursor");
        if (options.failAgentList && exactName === null) {
          return failure(
            503,
            "storage_transaction_conflict",
            "Catalog index unavailable",
          );
        }
        if (exactName === "researcher") {
          return cursor === "versions-next"
            ? json(page([agentTemplate("2")]))
            : json(page([agentTemplate("1")], "versions-next"));
        }
        if (options.paginated) {
          return json(
            page(
              [agentTemplate(cursor === "catalog-2" ? "2" : "1")],
              cursor === null
                ? "catalog-2"
                : cursor === "catalog-2"
                  ? "catalog-3"
                  : undefined,
            ),
          );
        }
        const query = (url.searchParams.get("q") ?? "").toLowerCase();
        if (query === options.slowSearch) {
          await aborted(request.signal);
          abortedSearches.push(query);
        }
        return json(
          page(
            [agentTemplate("1"), agentTemplate("2")].filter((item) =>
              `${item.ref.name} ${item.ref.version} ${description}`
                .toLowerCase()
                .includes(query),
            ),
          ),
        );
      },
      "GET /v1/workflows/{name}/versions/{version}": ({ params }) =>
        json(workflowResource(params.name ?? "", params.version)),
    },
    emptyPageOrNotFound,
  );
  const view = renderApplication(path, { server, capabilities: ["user"] });
  currentRouter = view.router;
  return { abortedSearches, ...view };
}

/** Whether `server` received a request whose query has q=`value`. */
function searched(server: { requests: Request[] }, value: string): boolean {
  return server.requests.some(
    (request) => new URL(request.url).searchParams.get("q") === value,
  );
}

async function currentPane() {
  return within(
    await screen.findByRole("region", {
      name:
        currentRouter.state.location.pathname !== "/catalog/agents"
          ? "Where used"
          : "Agents",
    }),
  );
}
async function paging() {
  return within(
    await screen.findByRole("navigation", {
      name:
        currentRouter.state.location.pathname !== "/catalog/agents"
          ? "Agent usage pages"
          : "Agent pages",
    }),
  );
}

describe("Catalog", () => {
  it("keeps discovery and usage cursor histories independent in the persistent workspace", async () => {
    const { router } = setup("/catalog/agents?q=researcher", {
      paginated: true,
      paginatedUsage: true,
    });
    const user = userEvent.setup();
    const list = within(await screen.findByRole("region", { name: "Agents" }));
    const listNode = screen.getByRole("region", { name: "Agents" });
    await user.click(await list.findByRole("button", { name: "Next" }));
    await list.findByText("Page 2");
    await user.click(
      await list.findByRole("link", { name: /Model agent Researcher/ }),
    );
    expect(screen.getByRole("region", { name: "Agents" })).toBe(listNode);
    const usage = within(
      await screen.findByRole("region", { name: "Where used" }),
    );
    await user.click(await usage.findByRole("button", { name: "Next" }));
    await usage.findByText("Page 2");
    expect(
      new URLSearchParams(router.state.location.search).get("cursor"),
    ).toBe("catalog-2");
    expect(
      new URLSearchParams(router.state.location.search).get("usageCursor"),
    ).toBe("usage-next");
    await user.click(list.getByRole("button", { name: "Previous" }));
    await list.findByText("Page 1");
    expect(usage.getByRole("button", { name: "Previous" })).toBeEnabled();
    await user.click(usage.getByRole("button", { name: "Previous" }));
    await usage.findByText("Page 1");
    expect(router.state.location.search).toBe("?q=researcher");
  });

  it.each(["agents", "audit-presets", "workflows"])(
    "does not substitute a default for an incomplete exact %s URL",
    async (section) => {
      const { sent } = setup(`/catalog/${section}/named-without-version`);
      expect(
        await screen.findByText(/(version|route) is invalid/),
      ).toBeVisible();
      expect(
        sent("GET", `/v1/${section}/named-without-version/versions/1`),
      ).toHaveLength(0);
    },
  );

  it("learns only visited predecessors after opening a later page directly", async () => {
    const user = userEvent.setup();
    const { router } = setup(
      "/catalog/agents?q=researcher&cursor=catalog-2&page=4",
      { paginated: true },
    );
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
    await user.click((await paging()).getByRole("button", { name: "Next" }));
    await (await currentPane()).findByText("Page 5");
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeEnabled();
    await user.click(
      (await paging()).getByRole("button", { name: "Previous" }),
    );
    await (await currentPane()).findByText("Page 4");
    expect(router.state.location.search).toBe(
      "?q=researcher&cursor=catalog-2&page=4",
    );
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
  });

  it("does not reuse cursor history for changed filters or another Agent context", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/agents?q=researcher", {
      paginated: true,
      paginatedUsage: true,
    });
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 2");
    const previousState = router.state.location.state;
    await act(async () => {
      await router.navigate(
        "/catalog/agents?q=changed&cursor=catalog-2&page=2",
        { state: previousState },
      );
    });
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
    await act(async () => {
      await router.navigate(
        "/catalog/agents/researcher/1?usageCursor=usage-next&usagePage=2",
      );
    });
    await user.click(
      await (await paging()).findByRole("button", { name: "First page" }),
    );
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 2");
    const usageState = router.state.location.state;
    await act(async () => {
      await router.navigate(
        "/catalog/agents/researcher/2?usageCursor=usage-next&usagePage=2",
        { state: usageState },
      );
    });
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
    expect(
      (await paging()).getByRole("button", { name: "First page" }),
    ).toBeEnabled();
  });

  it("returns through actual preceding Agent pages after a detail visit and browser navigation", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/agents?q=researcher&keep=yes", {
      paginated: true,
    });
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 2");
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 3");
    await user.click(
      await screen.findByRole("link", { name: /Model agent Researcher/ }),
    );
    await user.click(await screen.findByRole("link", { name: "All agents" }));
    await (await currentPane()).findByText("Page 3");
    await user.click(
      await (await paging()).findByRole("button", { name: "Previous" }),
    );
    expect(router.state.location.pathname).toBe("/catalog/agents");
    expect(
      new URLSearchParams(router.state.location.search).get("cursor"),
    ).toBe("catalog-2");
    await (await currentPane()).findByText("Page 2");
    await user.click(
      await (await paging()).findByRole("button", { name: "Previous" }),
    );
    await (await currentPane()).findByText("Page 1");
    expect(router.state.location.search).toBe("?q=researcher&keep=yes");
    await act(async () => {
      await router.navigate(-1);
    });
    await (await currentPane()).findByText("Page 2");
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeEnabled();
    await act(async () => {
      await router.navigate(1);
    });
    await (await currentPane()).findByText("Page 1");
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
  });

  it("disables unknown predecessors on direct links and offers the same query's first page", async () => {
    const user = userEvent.setup();
    const { router } = setup(
      "/catalog/agents?q=researcher&keep=yes&cursor=external&page=4",
      { paginated: true },
    );
    await (await currentPane()).findByText("Page 4");
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
    await user.click(
      (await paging()).getByRole("button", { name: "First page" }),
    );
    await (await currentPane()).findByText("Page 1");
    expect(router.state.location.search).toBe("?q=researcher&keep=yes");
  });

  it("resets cursor history when the search changes", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/agents?keep=yes", { paginated: true });
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 2");
    await user.type(screen.getByLabelText("Search agents"), "researcher");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?keep=yes&q=researcher"),
    );
    await waitFor(async () =>
      expect(
        (await paging()).getByRole("button", { name: "Previous" }),
      ).toBeDisabled(),
    );
    await user.click((await paging()).getByRole("button", { name: "Next" }));
    await (await currentPane()).findByText("Page 2");
    await user.click(
      await (await paging()).findByRole("button", { name: "Previous" }),
    );
    expect(router.state.location.search).toBe("?keep=yes&q=researcher");
  });

  it("keeps the search text as typed across a debounced commit", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/agents?keep=yes");
    const search = await screen.findByLabelText("Search agents");
    await user.type(search, "supplied ");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?keep=yes&q=supplied"),
    );
    expect(search).toHaveValue("supplied ");

    await user.type(search, "evidence");
    expect(search).toHaveValue("supplied evidence");
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?keep=yes&q=supplied+evidence",
      ),
    );
    expect(search).toHaveValue("supplied evidence");

    // Navigation that clears the query also replaces the committed draft
    // instead of reviving and re-committing it.
    await act(async () => {
      await router.navigate("/catalog/agents?keep=yes");
    });
    expect(search).toHaveValue("");
    await act(async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 400));
    });
    expect(router.state.location.search).toBe("?keep=yes");
    expect(search).toHaveValue("");
  });

  it("preserves Agent usage cursors through Workflow visits and resets them for another version", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/agents/researcher/1", {
      paginatedUsage: true,
    });
    const usage = () =>
      within(screen.getByRole("region", { name: "Where used" }));
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 2");
    await user.click(
      await usage().findByRole("link", { name: /openapi-from-source@1/ }),
    );
    await user.click(await screen.findByRole("link", { name: /researcher@1/ }));
    await (await currentPane()).findByText("Page 2");
    await user.click(
      await (await paging()).findByRole("button", { name: "Previous" }),
    );
    expect(router.state.location.pathname).toBe("/catalog/agents/researcher/1");
    expect(router.state.location.search).toBe("");
    await user.click(
      await (await paging()).findByRole("button", { name: "Next" }),
    );
    await (await currentPane()).findByText("Page 2");
    await user.click(
      screen.getByRole("button", { name: "Load more versions" }),
    );
    await screen.findByRole("option", { name: "2" });
    await user.selectOptions(screen.getByLabelText("Published version"), "2");
    await (await currentPane()).findByText("Page 1");
    expect(router.state.location.pathname).toBe("/catalog/agents/researcher/2");
    expect(router.state.location.search).toBe("");
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
  });

  it("offers the first usage page when an Agent usage URL is opened directly", async () => {
    const user = userEvent.setup();
    const { router } = setup(
      "/catalog/agents/researcher/1?usageCursor=external&usagePage=4",
      { paginatedUsage: true },
    );
    await (await currentPane()).findByText("Page 4");
    expect(
      await (await paging()).findByRole("button", { name: "Previous" }),
    ).toBeDisabled();
    await user.click(
      (await paging()).getByRole("button", { name: "First page" }),
    );
    await (await currentPane()).findByText("Page 1");
    expect(router.state.location.search).toBe("");
  });

  it("groups Agent versions and preserves a deliberate older choice for ordinary users", async () => {
    setup("/catalog/agents");
    const links = await screen.findAllByRole("link", {
      name: /Model agent Researcher/,
    });
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/catalog/agents/researcher/2",
    ]);
    await screen.findByRole("option", { name: "1" });
    await userEvent
      .setup()
      .selectOptions(screen.getByLabelText("Published version"), "1");
    expect(
      screen.getByRole("link", { name: /Model agent Researcher/ }),
    ).toHaveAttribute("href", "/catalog/agents/researcher/1");
    const primary = screen.getByRole("navigation", {
      name: "Primary navigation",
    });
    expect(
      within(primary).getByRole("link", { name: "Library" }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(primary).queryByRole("link", { name: "Skills" }),
    ).not.toBeInTheDocument();
    expect(
      within(
        screen.getByRole("navigation", { name: "Library sections" }),
      ).getByRole("link", { name: "Agents" }),
    ).toHaveAttribute("aria-current", "page");
  });

  it("opens Check types first and keeps the query and hash", async () => {
    const { router } = setup("/catalog?keep=yes#section");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/catalog/audit-presets"),
    );
    expect(router.state.location.search).toBe("?keep=yes");
    expect(router.state.location.hash).toBe("#section");
    expect(
      await screen.findByRole("heading", { level: 1, name: "Library" }),
    ).toBeVisible();
    expect(
      await screen.findByRole("heading", {
        level: 2,
        name: "Find your next check",
      }),
    ).toBeVisible();
    await waitFor(() =>
      expect(document.title).toBe("Check types · Library · Contractor"),
    );
  });

  it("shows the Library tabs with Files on every section", async () => {
    const user = userEvent.setup();
    const { router } = setup("/catalog/agents");
    const tabs = await screen.findByRole("navigation", {
      name: "Library sections",
    });
    expect(
      within(tabs)
        .getAllByRole("link")
        .map((link) => [link.textContent, link.getAttribute("href")]),
    ).toEqual([
      ["Check types", "/catalog/audit-presets"],
      ["Workflows", "/catalog/workflows"],
      ["Agents", "/catalog/agents"],
      ["Skills", "/catalog/skills"],
      ["Files", "/artifacts"],
    ]);
    await waitFor(() =>
      expect(document.title).toBe("Agents · Library · Contractor"),
    );
    await user.click(within(tabs).getByRole("link", { name: "Workflows" }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/catalog/workflows"),
    );
    expect(
      await screen.findByRole("heading", { level: 2, name: "Workflows" }),
    ).toBeVisible();
    expect(
      within(
        screen.getByRole("navigation", { name: "Library sections" }),
      ).getByRole("link", { name: "Workflows" }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(
        screen.getByRole("navigation", { name: "Primary navigation" }),
      ).getByRole("link", { name: "Library" }),
    ).toHaveAttribute("aria-current", "page");
    await waitFor(() =>
      expect(document.title).toBe("Workflows · Library · Contractor"),
    );
  });

  it("searches the complete Agent catalog on the server and preserves the query on return", async () => {
    const user = userEvent.setup();
    const { server, router } = setup("/catalog/agents?keep=yes");
    await screen.findAllByText("Researches the supplied evidence");
    await user.type(screen.getByLabelText("Search agents"), "2");
    await waitFor(() => expect(router.state.location.search).toContain("q=2"));
    await waitFor(() => expect(searched(server, "2")).toBe(true));
    const result = await screen.findByRole("link", {
      name: /Model agent Researcher/,
    });
    expect(
      screen.getAllByRole("link", { name: /Model agent Researcher/ }),
    ).toHaveLength(1);
    expect(result).toHaveAttribute(
      "href",
      "/catalog/agents/researcher/2?keep=yes&q=2",
    );
    await user.click(result);
    await screen.findByRole("heading", { name: "Second version" });
    await user.click(screen.getByRole("link", { name: "All agents" }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/catalog/agents"),
    );
    expect(router.state.location.search).toContain("keep=yes");
    expect(router.state.location.search).toContain("q=2");
  });

  it("cancels a superseded Agent search before showing the current query", async () => {
    const user = userEvent.setup();
    const { abortedSearches, server } = setup("/catalog/agents", {
      slowSearch: "slow",
    });
    await screen.findAllByText("Researches the supplied evidence");
    const search = screen.getByLabelText("Search agents");
    await user.type(search, "slow");
    await waitFor(() => expect(searched(server, "slow")).toBe(true));
    await user.clear(search);
    await user.type(search, "2");
    await screen.findByRole("link", {
      name: /Model agent Researcher/,
    });
    await waitFor(() => expect(abortedSearches).toEqual(["slow"]));
  });

  it("distinguishes empty Agent search results from catalog failures", async () => {
    const empty = setup("/catalog/agents?q=absent");
    expect(
      await screen.findByText("No Agents match this search."),
    ).toBeVisible();
    empty.unmount();
    setup("/catalog/agents", { failAgentList: true });
    expect(await screen.findByText("Catalog index unavailable")).toBeVisible();
  });

  it("previews safe Markdown, copies literal source and opens an exact version", async () => {
    const user = userEvent.setup();
    const copy = vi.spyOn(navigator.clipboard, "writeText");
    const { router, container } = setup("/catalog/agents/researcher/1");
    expect(
      await screen.findByRole("heading", { name: "Careful researcher" }),
    ).toBeVisible();
    expect(container.querySelector("script")).toBeNull();
    expect(
      container.querySelector('img[src="https://example.test/image.png"]'),
    ).toBeNull();
    expect(
      within(
        screen.getByRole("complementary", { name: "Agent capabilities" }),
      ).getByRole("link", { name: /skills\/research/ }),
    ).toHaveAttribute("href", "/artifacts/skills/research");
    await user.click(screen.getByRole("button", { name: "Copy instructions" }));
    expect(copy).toHaveBeenCalledWith(prompt);
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Prompt copied.",
    );
    await user.click(screen.getByRole("button", { name: "Source" }));
    expect(
      container.querySelector("pre.catalog-prompt-source")?.textContent,
    ).toBe(prompt);
    expect(
      within(screen.getByRole("region", { name: "Where used" })).getByText(
        /Stage analyze/,
      ),
    ).toBeVisible();
    expect(screen.queryByRole("option", { name: "2" })).not.toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Load more versions" }),
    );
    await screen.findByRole("option", { name: "2" });
    await user.selectOptions(screen.getByLabelText("Published version"), "2");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/catalog/agents/researcher/2",
      ),
    );
    expect(await screen.findByText(/Second version/)).toBeVisible();
    const usage = screen.getByRole("region", { name: "Where used" });
    expect(within(usage).getByText(/Stage model/)).toBeVisible();
    expect(
      within(usage).getByRole("link", { name: /likec4-from-source@beta/ }),
    ).toHaveAttribute("href", "/catalog/workflows/likec4-from-source/beta");
    await user.click(
      within(usage).getByRole("link", { name: /likec4-from-source@beta/ }),
    );
    expect(
      await screen.findByRole("heading", { name: "LikeC4 from source" }),
    ).toBeVisible();
    await user.click(screen.getByRole("link", { name: /researcher@2/ }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/catalog/agents/researcher/2",
      ),
    );
    expect(screen.getByLabelText("Published version")).toHaveValue("2");
  });

  it("reports clipboard failures without discarding the source", async () => {
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("denied"),
    );
    setup("/catalog/agents/researcher/1");
    await screen.findByRole("heading", { name: "Careful researcher" });
    await user.click(screen.getByRole("button", { name: "Copy instructions" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Could not copy",
    );
    await user.click(screen.getByRole("button", { name: "Source" }));
    expect(
      document.querySelector("pre.catalog-prompt-source")?.textContent,
    ).toBe(prompt);
  });

  it.each([
    "/skills",
    "/workflows",
    "/workflows/openapi-from-source/1",
    "/queue",
    "/operations/runtime-configs",
    "/operations/runtime-configs/debug/1",
  ])("shows recovery for retired route %s", async (path) => {
    const { router } = setup(`${path}?keep=yes#section`);
    expect(
      await screen.findByRole("heading", { name: "Page not found" }),
    ).toBeVisible();
    expect(router.state.location.pathname).toBe(path);
    expect(router.state.location.search).toBe("?keep=yes");
    expect(router.state.location.hash).toBe("#section");
    expect(screen.getByRole("link", { name: "Projects" })).toHaveAttribute(
      "href",
      "/projects",
    );
  });

  it("shows a tool agent's configuration without a prompt", async () => {
    const { server } = setup("/catalog/agents/researcher/1", {
      toolAgent: true,
    });
    expect(
      await screen.findByText(
        "This agent runs its configured tool directly without a model or prompt.",
      ),
    ).toBeVisible();
    const configuration = screen.getByRole("complementary", {
      name: "Agent capabilities",
    });
    expect(within(configuration).getByText("tool@1")).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Copy instructions" }),
    ).not.toBeInTheDocument();
    expect(
      await within(
        screen.getByRole("region", { name: "Where used" }),
      ).findByRole("link", { name: /openapi-from-source@1/ }),
    ).toBeVisible();
    expect(
      server.requests.some((request) => request.url.includes("/instructions")),
    ).toBe(false);
  });

  it("does not display or copy instructions from another digest", async () => {
    setup("/catalog/agents/researcher/1", { mismatch: true });
    expect(await screen.findByText(/instructions do not match/)).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Copy instructions" }),
    ).toBeDisabled();
    expect(
      screen.queryByRole("heading", { name: "Careful researcher" }),
    ).not.toBeInTheDocument();
  });
});
