import { fireEvent, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { components } from "../../api/generated/public";
import { conforming, DIGEST } from "../../test/builders";
import { failure, fakeServer, json, page } from "../../test/fake-server";
import { renderWithAPI } from "../../test/render";
import { CatalogField, type CatalogKind } from "./catalog-picker";

type Schemas = components["schemas"];

function agent(name: string, version: string) {
  return conforming("ConfigurationResource", {
    ref: { kind: "agent-templates", name, version, digest: DIGEST },
    source: "operator",
    body: {
      description: `Published ${name}`,
      runtime: "llm@1",
      instructions: { ref: "instructions/agent.md", digest: DIGEST },
      modelPolicy: { policyId: "worker", version: "1", digest: DIGEST },
      sandboxProfile: "default@1",
      toolsets: [],
    },
  });
}

/** Version 3 of a published configuration named "policy". */
function configuration(
  kind: Schemas["ConfigurationRef"]["kind"],
  body: Schemas["ConfigurationResource"]["body"],
) {
  const ref = { kind, name: "policy", version: "3", digest: DIGEST };
  return conforming("ConfigurationResource", { ref, source: "operator", body });
}

/** One published version of each other catalog kind. */
const published = {
  workflows: () =>
    conforming("WorkflowSummary", {
      ref: { name: "workflow", version: "5" },
      entryStage: "start",
      inputs: {},
      outputs: {},
      parameters: {},
      presentation: {
        displayName: "Workflow",
        description: "Published workflow",
      },
    }),
  "model-policies": () =>
    configuration("model-policies", {
      model: "test-model",
      contextWindowTokens: 1000,
    }),
  "llm-gateways": () =>
    configuration("llm-gateways", {
      protocol: "openai-compatible@1",
      url: "http://gateway.test/v1",
    }),
  "execution-configs": () =>
    configuration("execution-configs", {
      planner: {
        modelPolicy: { policyId: "worker", version: "1", digest: DIGEST },
      },
    }),
};

const failed = () => failure(503, "unavailable", "Try again", true);

/** The field over a fake Server whose catalog lists answer `load`. */
function mount(
  load: (url: URL) => Response,
  kind: CatalogKind = "agent-templates",
) {
  const answer = ({ url }: { url: URL }) => load(url);
  const change = vi.fn();
  const view = renderWithAPI(
    <CatalogField
      label="Template"
      kind={kind}
      value="manual@7"
      onChange={change}
    />,
    {
      server: fakeServer({
        "GET /v1/configurations/{kind}": answer,
        "GET /v1/workflows": answer,
      }),
    },
  );
  return { change, requests: view.requests, user: view.user };
}

describe("Studio catalog selectors", () => {
  it("loads only when opened, cancels without edits and chooses an exact published version", async () => {
    const { change, requests, user } = mount(() =>
      json(page([agent("worker", "1"), agent("worker", "2")])),
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
    let fail = true;
    const { requests, user } = mount((url) => {
      if (url.searchParams.get("q") === "target")
        return json(page([agent("target", "3")]));
      if (url.searchParams.has("cursor")) {
        if (fail) {
          fail = false;
          return failed();
        }
        return json(page([agent("worker", "1"), agent("second", "2")]));
      }
      return json(page([agent("worker", "1")], "page-two"));
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
    const { change, user } = mount(() => failed());
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
      const { requests, user } = mount(() =>
        json({
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
    const { change, requests, user } = mount(
      () => json(page([published[kind]()])),
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
