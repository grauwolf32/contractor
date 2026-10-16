import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { renderApplication } from "../../../test/application";
import { operationsSnapshot } from "../../../test/builders-execution";
import { json, page } from "../../../test/responses";

const digest = `sha256:${"5".repeat(64)}`;

const gateway = {
  ref: { kind: "llm-gateways", name: "local-litellm", version: "1", digest },
  body: { protocol: "openai-compatible@1", url: "http://127.0.0.1:4000/v1" },
  source: "operator",
};
const policy = {
  ref: { kind: "model-policies", name: "worker", version: "1", digest },
  body: { model: "qwen-worker" },
  source: "managed",
};

function setup(path: string) {
  const reads: string[] = [];
  const { router } = renderApplication(path, {
    server: {
      "GET /v1/operations/snapshot": () => json(operationsSnapshot("1")),
      "GET /v1/projects": () => json(page([])),
      "GET /v1/configurations/{kind}": ({ url, params }) => {
        reads.push(url.pathname);
        if (params.kind === "llm-gateways") return json(page([gateway]));
        if (params.kind === "model-policies") return json(page([policy]));
        return json(page([]));
      },
      "GET /v1/configurations/{kind}/{name}/versions/{version}": ({ url }) => {
        reads.push(url.pathname);
        return json(gateway);
      },
    },
  });
  return { router, reads };
}

describe("LLM configurations", () => {
  it("keeps the chosen kind in the address and lists that kind", async () => {
    const { router, reads } = setup(
      "/operations/configurations?kind=llm-gateways",
    );
    expect(
      await screen.findByText("local-litellm@1", { selector: "strong" }),
    ).toBeInTheDocument();
    const kinds = screen.getByRole("group", { name: "Kind" });
    expect(
      within(kinds)
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual([
      "Model policies",
      "LLM gateways",
      "Agent templates",
      "Execution configs",
    ]);
    expect(
      within(kinds).getByRole("button", { name: "LLM gateways" }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(reads).toEqual(["/v1/configurations/llm-gateways"]);
    expect(
      screen.getByRole("link", { name: "Inspect / clone" }),
    ).toHaveAttribute(
      "href",
      "/operations/configurations/llm-gateways/local-litellm/1",
    );

    await userEvent
      .setup()
      .click(within(kinds).getByRole("button", { name: "Model policies" }));
    expect(
      await screen.findByText("worker@1", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(router.state.location.search).toBe("");
    expect(reads).toContain("/v1/configurations/model-policies");
  });

  it("leads back to the list of the same kind from a version", async () => {
    setup("/operations/configurations/llm-gateways/local-litellm/1");
    expect(
      await screen.findByRole("heading", { name: "Published values" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "local-litellm@1" }),
    ).toBeInTheDocument();
    const breadcrumb = screen.getByRole("navigation", { name: "Breadcrumb" });
    expect(
      within(breadcrumb).getByRole("link", { name: "LLM gateways" }),
    ).toHaveAttribute("href", "/operations/configurations?kind=llm-gateways");
    expect(
      within(breadcrumb).getByRole("link", { name: "LLM configurations" }),
    ).toHaveAttribute("href", "/operations/configurations");
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Clone to new version" }),
      ).toBeEnabled(),
    );
  });
});
