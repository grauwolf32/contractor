import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";

const digest = `sha256:${"1".repeat(64)}`;
const secondDigest = `sha256:${"2".repeat(64)}`;
const prompt =
  "# Careful researcher\n\nRead the evidence first.\n\n![remote](https://example.test/image.png)\n<script>alert('no')</script>";
const session: AuthSession = {
  principal: { userId: "owner", username: "owner", capabilities: ["user"] },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-10T00:00:00Z",
  absoluteExpiresAt: "2026-09-11T00:00:00Z",
};
const configuration = (version = "1") => {
  const exactDigest = version === "2" ? secondDigest : digest;
  return {
    ref: {
      kind: "agent-templates",
      name: "researcher",
      version,
      digest: exactDigest,
    },
    source: "operator",
    body: {
      description: "Researches the supplied evidence",
      runtime: "adk@1",
      instructions: {
        ref: "instructions/researcher.md",
        digest: exactDigest,
      },
      modelPolicy: { policyId: "worker", version: "1", digest },
      toolsets: [{ ref: "run-artifacts@1", tools: ["read_artifact"] }],
      skills: [{ namespace: "skills", name: "research" }],
      sandboxProfile: "local-workdir@1",
    },
  };
};

function setup(
  path: string,
  options: {
    mismatch?: boolean;
    failPrompt?: boolean;
    failAgentList?: boolean;
    slowSearch?: string;
  } = {},
) {
  const abortedSearches: string[] = [];
  const fetcher = vi.fn(async (request: RequestInfo | URL) => {
    const url = new URL(
      request instanceof Request ? request.url : String(request),
    );
    const version = url.pathname.includes("/versions/2") ? "2" : "1";
    const exactDigest = version === "2" ? secondDigest : digest;
    let body: unknown = { items: [], page: { hasMore: false } };
    let status = 200;
    if (url.pathname.endsWith("/workflow-bindings")) {
      body = {
        items: [
          version === "2"
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
        page: { hasMore: false },
      };
    } else if (url.pathname.endsWith("/instructions")) {
      body = {
        template: {
          templateId: "researcher",
          version,
          digest: options.mismatch ? secondDigest : exactDigest,
        },
        instructions: {
          ref: "instructions/researcher.md",
          digest: exactDigest,
          text: version === "2" ? "# Second version" : prompt,
        },
      };
      if (options.failPrompt) {
        body = {
          code: "not_found",
          message: "Instructions unavailable",
          retryable: false,
        };
        status = 404;
      }
    } else if (url.pathname.startsWith("/v1/workflows/")) {
      const parts = url.pathname.split("/");
      const workflowName = decodeURIComponent(parts[3] ?? "");
      const workflowVersion = decodeURIComponent(parts[5] ?? "");
      body = {
        ref: { name: workflowName, version: workflowVersion },
        entryStage: "analyze",
        parameters: {},
        inputs: {},
        outputs: {},
        stages: {},
      };
    } else if (url.pathname.includes("/versions/"))
      body = configuration(version);
    else if (url.pathname.endsWith("/agent-templates")) {
      if (options.failAgentList && url.searchParams.get("name") === null) {
        body = {
          code: "catalog_unavailable",
          message: "Catalog index unavailable",
          retryable: true,
        };
        status = 503;
      } else {
        const items = [configuration(), configuration("2")];
        const query = (url.searchParams.get("q") ?? "").toLowerCase();
        if (query === options.slowSearch) {
          const signal =
            request instanceof Request ? request.signal : undefined;
          await new Promise<void>((resolve) => {
            if (signal?.aborted) {
              abortedSearches.push(query);
              resolve();
              return;
            }
            signal?.addEventListener(
              "abort",
              () => {
                abortedSearches.push(query);
                resolve();
              },
              { once: true },
            );
          });
        }
        const exactName = url.searchParams.get("name");
        const versionCursor = url.searchParams.get("cursor");
        body = {
          items:
            exactName === "researcher"
              ? versionCursor === "versions-next"
                ? [items[1]]
                : [items[0]]
              : query === ""
                ? items
                : items.filter((item) =>
                    `${item.ref.name} ${item.ref.version} ${item.body.description}`
                      .toLowerCase()
                      .includes(query),
                  ),
          page:
            exactName === "researcher" && versionCursor === null
              ? { hasMore: true, nextCursor: "versions-next" }
              : { hasMore: false },
        };
      }
    }
    return new Response(JSON.stringify(body), {
      status,
      headers: {
        "Content-Type": "application/json",
        "X-Contractor-API-Version": "contractor.public.v1",
      },
    });
  });
  const publicAPI = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    fetcher,
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const view = render(
    <Application
      api={{
        getSession: async () => session,
        login: async () => session,
        logout: async () => undefined,
      }}
      publicAPI={publicAPI}
      router={router}
    />,
  );
  return { abortedSearches, fetcher, router, ...view };
}

describe("Catalog", () => {
  it("groups navigation and lists exact agent versions for ordinary users", async () => {
    setup("/catalog/agents");
    const links = await screen.findAllByRole("link", {
      name: /researcher.*Inspect exact version/,
    });
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/catalog/agents/researcher/1",
      "/catalog/agents/researcher/2",
    ]);
    const primary = screen.getByRole("navigation", {
      name: "Primary navigation",
    });
    expect(
      within(primary).getByRole("link", { name: "Catalog" }),
    ).toHaveAttribute("aria-current", "page");
    expect(
      within(primary).queryByRole("link", { name: "Skills" }),
    ).not.toBeInTheDocument();
    expect(
      within(
        screen.getByRole("navigation", { name: "Catalog navigation" }),
      ).getByRole("link", { name: "Agents" }),
    ).toHaveAttribute("aria-current", "page");
  });

  it("searches the complete Agent catalog on the server and preserves the query on return", async () => {
    const user = userEvent.setup();
    const { fetcher, router } = setup("/catalog/agents?keep=yes");
    await screen.findAllByText("Researches the supplied evidence");
    await user.type(screen.getByLabelText("Search agents"), "2");
    await waitFor(() => expect(router.state.location.search).toContain("q=2"));
    await waitFor(() =>
      expect(
        fetcher.mock.calls.some(([input]) => {
          const request = input instanceof Request ? input : new Request(input);
          return new URL(request.url).searchParams.get("q") === "2";
        }),
      ).toBe(true),
    );
    const result = await screen.findByRole("link", {
      name: /researcher.*2.*Inspect exact version/,
    });
    expect(
      screen.getAllByRole("link", { name: /Inspect exact version/ }),
    ).toHaveLength(1);
    expect(result).toHaveAttribute("href", "/catalog/agents/researcher/2");
    await user.click(result);
    await screen.findByRole("heading", { name: "Second version" });
    await user.click(screen.getByRole("link", { name: /Agent search/ }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/catalog/agents"),
    );
    expect(router.state.location.search).toContain("keep=yes");
    expect(router.state.location.search).toContain("q=2");
  });

  it("cancels a superseded Agent search before showing the current query", async () => {
    const user = userEvent.setup();
    const { abortedSearches, fetcher } = setup("/catalog/agents", {
      slowSearch: "slow",
    });
    await screen.findAllByText("Researches the supplied evidence");
    const search = screen.getByLabelText("Search agents");
    await user.type(search, "slow");
    await waitFor(() =>
      expect(
        fetcher.mock.calls.some(([input]) => {
          const request = input instanceof Request ? input : new Request(input);
          return new URL(request.url).searchParams.get("q") === "slow";
        }),
      ).toBe(true),
    );
    await user.clear(search);
    await user.type(search, "2");
    await screen.findByRole("link", {
      name: /researcher.*2.*Inspect exact version/,
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
      screen.getByRole("link", { name: "skills/research" }),
    ).toHaveAttribute("href", "/artifacts/skills/research");
    await user.click(screen.getByRole("button", { name: "Copy" }));
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
        "analyze",
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
    expect(within(usage).getByText("model")).toBeVisible();
    expect(
      within(usage).getByRole("link", { name: "likec4-from-source@beta" }),
    ).toHaveAttribute("href", "/catalog/workflows/likec4-from-source/beta");
    await user.click(
      within(usage).getByRole("link", { name: "likec4-from-source@beta" }),
    );
    expect(
      await screen.findByRole("heading", { name: "likec4-from-source" }),
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
    await user.click(screen.getByRole("button", { name: "Copy" }));
    expect(await screen.findByRole("status")).toHaveTextContent(
      "Could not copy",
    );
    await user.click(screen.getByRole("button", { name: "Source" }));
    expect(
      document.querySelector("pre.catalog-prompt-source")?.textContent,
    ).toBe(prompt);
  });

  it.each(["/skills", "/workflows"])(
    "preserves legacy %s query and fragment",
    async (path) => {
      const { router } = setup(`${path}?keep=yes#section`);
      await waitFor(() =>
        expect(router.state.location.pathname).toBe(`/catalog${path}`),
      );
      expect(router.state.location.search).toBe("?keep=yes");
      expect(router.state.location.hash).toBe("#section");
    },
  );

  it("does not display or copy instructions from another digest", async () => {
    setup("/catalog/agents/researcher/1", { mismatch: true });
    expect(await screen.findByText(/instructions do not match/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Copy" })).toBeDisabled();
    expect(
      screen.queryByRole("heading", { name: "Careful researcher" }),
    ).not.toBeInTheDocument();
  });
});
