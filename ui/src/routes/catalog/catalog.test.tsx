import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";

const digest = `sha256:${"1".repeat(64)}`;
const prompt =
  "# Careful researcher\n\nRead the evidence first.\n\n![remote](https://example.test/image.png)\n<script>alert('no')</script>";
const session: AuthSession = {
  principal: { userId: "owner", username: "owner", capabilities: ["user"] },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-10T00:00:00Z",
  absoluteExpiresAt: "2026-09-11T00:00:00Z",
};
const configuration = (version = "1") => ({
  ref: { kind: "agent-templates", name: "researcher", version, digest },
  source: "operator",
  body: {
    description: "Researches the supplied evidence",
    runtime: "adk@1",
    instructions: { ref: "instructions/researcher.md", digest },
    modelPolicy: { policyId: "worker", version: "1", digest },
    toolsets: [{ ref: "run-artifacts@1", tools: ["read_artifact"] }],
    skills: [{ namespace: "skills", name: "research" }],
    sandboxProfile: "local-workdir@1",
  },
});

function setup(
  path: string,
  options: { mismatch?: boolean; failPrompt?: boolean } = {},
) {
  const publicAPI = new PublicAPI(
    {
      uiVersion: "0.1.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    vi.fn(async (request: RequestInfo | URL) => {
      const url = new URL(
        request instanceof Request ? request.url : String(request),
      );
      const version = url.pathname.includes("/versions/2") ? "2" : "1";
      let body: unknown = { items: [], page: { hasMore: false } };
      let status = 200;
      if (url.pathname.endsWith("/instructions")) {
        body = {
          template: {
            templateId: "researcher",
            version,
            digest: options.mismatch ? `sha256:${"2".repeat(64)}` : digest,
          },
          instructions: {
            ref: "instructions/researcher.md",
            digest,
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
      } else if (url.pathname.includes("/versions/"))
        body = configuration(version);
      else if (url.pathname.endsWith("/agent-templates"))
        body = {
          items: [configuration(), configuration("2")],
          page: { hasMore: false },
        };
      return new Response(JSON.stringify(body), {
        status,
        headers: {
          "Content-Type": "application/json",
          "X-Contractor-API-Version": "contractor.public.v1",
        },
      });
    }),
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
  return { router, ...view };
}

describe("Catalog", () => {
  it("groups navigation and lists exact agent versions for ordinary users", async () => {
    setup("/catalog/agents");
    const links = await screen.findAllByRole("link", {
      name: /researcher.*View prompt/,
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
    await user.clear(screen.getByLabelText("Version"));
    await user.type(screen.getByLabelText("Version"), "2");
    await user.click(screen.getByRole("button", { name: "Open version" }));
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/catalog/agents/researcher/2",
      ),
    );
    expect(await screen.findByText(/Second version/)).toBeVisible();
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
