import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import { compareWorkflowVersions } from "./families";

const config = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};
const workflow = (version: string) => ({
  ref: { name: "check", version },
  entryStage: "check",
  presentation:
    version === "10"
      ? { displayName: "Source check", description: "Review source evidence." }
      : { displayName: "Source check" },
  parameters: {},
  inputs: {},
  outputs:
    version === "10"
      ? {
          report: {
            required: true,
            primary: true,
            mediaTypes: ["text/markdown"],
          },
        }
      : {},
});
function response(data: unknown) {
  return new Response(JSON.stringify(data), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

describe("shared Workflow cards", () => {
  it("orders numeric version components without lexical or number precision errors", () => {
    expect(compareWorkflowVersions("10", "9")).toBeGreaterThan(0);
    expect(compareWorkflowVersions("1.10", "1.9")).toBeGreaterThan(0);
    expect(
      compareWorkflowVersions("999999999999999999999", "999999999999999999998"),
    ).toBeGreaterThan(0);
  });
  it("defaults to the latest version across pages, pins an older choice on refresh and keeps exact links and contracts together", async () => {
    let secondRead = false;
    const api = new PublicAPI(
      config,
      vi.fn(async (input) => {
        const request = input instanceof Request ? input : new Request(input);
        const url = new URL(request.url);
        if (url.pathname === "/v1/auth/session")
          return response({
            principal: {
              userId: "user",
              username: "owner",
              capabilities: ["user"],
            },
            csrfToken: "a".repeat(43),
            idleExpiresAt: "2099-01-01T00:00:00Z",
            absoluteExpiresAt: "2099-01-02T00:00:00Z",
          });
        if (url.pathname === "/v1/workflows") {
          if (url.searchParams.get("cursor") === "next") {
            secondRead = true;
            return response({
              items: [workflow("10")],
              page: { hasMore: false },
            });
          }
          return response({
            items: [workflow("9")],
            page: { hasMore: true, nextCursor: "next" },
          });
        }
        throw Error(url.pathname);
      }),
    );
    const router = createMemoryRouter(applicationRoutes(), {
      initialEntries: ["/catalog/workflows"],
    });
    render(<Application api={api} publicAPI={api} router={router} />);
    const card = await screen.findByRole("article", { name: "Source check" });
    expect(secondRead).toBe(true);
    expect(screen.getAllByRole("article")).toHaveLength(1);
    const select = within(card).getByRole("combobox", {
      name: "Version of check",
    });
    expect(select).toHaveValue("10");
    expect(
      within(card).getByRole("link", { name: "View workflow" }),
    ).toHaveAttribute("href", "/catalog/workflows/check/10");
    expect(within(card).getByText("report")).toBeVisible();
    expect(
      within(card).getByText("Review source evidence."),
    ).toBeInTheDocument();
    const user = userEvent.setup();
    await user.selectOptions(select, "9");
    expect(within(card).queryByText("report")).toBeNull();
    // A version without a description simply omits the purpose line.
    expect(within(card).queryByText("Review source evidence.")).toBeNull();
    expect(within(card).queryByText(/Purpose is not described/)).toBeNull();
    expect(card.querySelector(".workflow-card-description")).toBeNull();
    expect(
      within(card).getByRole("link", { name: "View workflow" }),
    ).toHaveAttribute("href", "/catalog/workflows/check/9");
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled(),
    );
    expect(select).toHaveValue("9");
  });
});
