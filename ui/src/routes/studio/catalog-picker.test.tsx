import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../api/client";
import { PublicAPIProvider } from "../../api/context";
import { CatalogField, type CatalogKind } from "./catalog-picker";

function response(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "Content-Type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}
function agent(name: string, version: string) {
  return {
    ref: {
      kind: "agent-templates",
      name,
      version,
      digest: `sha256:${"a".repeat(64)}`,
    },
    source: "operator",
    body: {
      description: `Published ${name}`,
      runtime: "llm@1",
      sandboxProfile: "default@1",
      toolsets: [],
    },
  };
}
const failed = () =>
  response(
    {
      code: "unavailable",
      message: "Try again",
      requestId: "test",
      retryable: true,
    },
    503,
  );
function mount(
  load: (url: URL) => Promise<Response>,
  kind: CatalogKind = "agent-templates",
) {
  const requests: Request[] = [];
  const api = new PublicAPI(
    {
      uiVersion: "0.6.0",
      supportedApiVersions: ["contractor.public.v1"],
      apiBaseUrl: "http://127.0.0.1:8080",
    },
    async (input) => {
      const request = input as Request;
      requests.push(request);
      return load(new URL(request.url));
    },
  );
  const change = vi.fn();
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <PublicAPIProvider api={api}>
        <CatalogField
          label="Template"
          kind={kind}
          value="manual@7"
          onChange={change}
        />
      </PublicAPIProvider>
    </QueryClientProvider>,
  );
  return { change, requests };
}
describe("Studio catalog selectors", () => {
  it("loads only when opened, cancels without edits and chooses an exact published version", async () => {
    const user = userEvent.setup();
    const { change, requests } = mount(async () =>
      response({
        items: [agent("worker", "1"), agent("worker", "2")],
        page: { hasMore: false },
      }),
    );
    expect(requests).toHaveLength(0);
    const open = screen.getByRole("button", {
      name: "Choose template from catalog",
    });
    await user.click(open);
    await screen.findByRole("button", { name: "Use worker@2" });
    expect(screen.getByLabelText("Search catalog")).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(change).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Template")).toHaveValue("manual@7");
    expect(open).toHaveFocus();
    await user.click(open);
    await user.click(
      await screen.findByRole("button", { name: "Use worker@2" }),
    );
    expect(change).toHaveBeenCalledExactlyOnceWith("worker@2");
    expect(requests.every((request) => request.method === "GET")).toBe(true);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
  it("retains loaded choices on continuation failure, retries, deduplicates and resets cursors on search", async () => {
    const user = userEvent.setup();
    let fail = true;
    const { requests } = mount(async (url) => {
      if (url.searchParams.get("q") === "target")
        return response({
          items: [agent("target", "3")],
          page: { hasMore: false },
        });
      if (url.searchParams.has("cursor")) {
        if (fail) {
          fail = false;
          return failed();
        }
        return response({
          items: [agent("worker", "1"), agent("second", "2")],
          page: { hasMore: false },
        });
      }
      return response({
        items: [agent("worker", "1")],
        page: { hasMore: true, nextCursor: "page-two" },
      });
    });
    await user.click(
      screen.getByRole("button", { name: "Choose template from catalog" }),
    );
    await user.click(
      await screen.findByRole("button", { name: "Load more versions" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "More versions could not be loaded",
    );
    expect(screen.getByRole("button", { name: "Use worker@1" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Retry catalog" }));
    await screen.findByRole("button", { name: "Use second@2" });
    expect(
      screen.getAllByRole("button", { name: "Use worker@1" }),
    ).toHaveLength(1);
    await user.type(screen.getByLabelText("Search catalog"), "target");
    await user.click(screen.getByRole("button", { name: "Search" }));
    await screen.findByRole("button", { name: "Use target@3" });
    expect(
      screen.queryByRole("button", { name: "Use worker@1" }),
    ).not.toBeInTheDocument();
    const last = new URL(requests.at(-1)!.url);
    expect(last.searchParams.get("q")).toBe("target");
    expect(last.searchParams.has("cursor")).toBe(false);
    expect(last.searchParams.get("limit")).toBe("50");
  });
  it("leaves manual editing available after an initial catalog failure", async () => {
    const user = userEvent.setup();
    const { change } = mount(async () => failed());
    await user.click(
      screen.getByRole("button", { name: "Choose template from catalog" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Catalog could not be loaded",
    );
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    fireEvent.change(screen.getByLabelText("Template"), {
      target: { value: "offline@4" },
    });
    fireEvent.blur(screen.getByLabelText("Template"));
    expect(change).toHaveBeenCalledExactlyOnceWith("offline@4");
  });
  it.each(["repeated", "missing"])(
    "stops a %s continuation without hiding loaded choices",
    async (condition) => {
      const user = userEvent.setup();
      const { requests } = mount(async () =>
        response({
          items: [agent("worker", "1")],
          page: {
            hasMore: true,
            ...(condition === "repeated" ? { nextCursor: "repeat" } : {}),
          },
        }),
      );
      await user.click(
        screen.getByRole("button", { name: "Choose template from catalog" }),
      );
      await screen.findByRole("button", { name: "Use worker@1" });
      if (condition === "repeated")
        await user.click(
          screen.getByRole("button", { name: "Load more versions" }),
        );
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Narrow the search",
      );
      expect(
        screen.queryByRole("button", { name: "Load more versions" }),
      ).not.toBeInTheDocument();
      expect(requests).toHaveLength(condition === "repeated" ? 2 : 1);
      expect(
        screen.getByRole("button", { name: "Use worker@1" }),
      ).toBeVisible();
    },
  );
  it.each([
    "workflows",
    "model-policies",
    "llm-gateways",
    "execution-configs",
  ] as const)("reads published %s through its own endpoint", async (kind) => {
    const user = userEvent.setup();
    const { change, requests } = mount(
      async () =>
        response({
          items: [
            kind === "workflows"
              ? {
                  ref: {
                    name: "workflow",
                    version: "5",
                    digest: `sha256:${"a".repeat(64)}`,
                  },
                  entryStage: "start",
                  inputs: {},
                  outputs: {},
                  parameters: {},
                  presentation: {
                    displayName: "Workflow",
                    description: "Published workflow",
                  },
                }
              : {
                  ref: {
                    kind,
                    name: "policy",
                    version: "3",
                    digest: `sha256:${"a".repeat(64)}`,
                  },
                  source: "operator",
                  body: { model: "test-model", contextWindowTokens: 1000 },
                },
          ],
          page: { hasMore: false },
        }),
      kind,
    );
    await user.click(
      screen.getByRole("button", { name: "Choose template from catalog" }),
    );
    const dialog = await screen.findByRole("dialog");
    const selector = kind === "workflows" ? "workflow@5" : "policy@3";
    await user.click(
      await within(dialog).findByRole("button", { name: `Use ${selector}` }),
    );
    expect(change).toHaveBeenCalledExactlyOnceWith(selector);
    expect(new URL(requests[0]!.url).pathname).toBe(
      kind === "workflows" ? "/v1/workflows" : `/v1/configurations/${kind}`,
    );
  });
});
