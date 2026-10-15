import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  workflowResource,
  workflowSummary,
} from "../../test/builders-execution";
import { json, page } from "../../test/fake-server";
import { renderApplication } from "../artifacts/test-support";

const workflow = (version: string) =>
  workflowSummary(
    workflowResource("check", version, {
      entryStage: "check",
      // An older version has no presentation (name and purpose come together).
      ...(version === "10"
        ? {
            presentation: {
              displayName: "Source check",
              description: "Review source evidence.",
            },
            outputs: {
              report: {
                required: true,
                primary: true,
                mediaTypes: ["text/markdown"],
              },
            },
          }
        : {}),
    }),
  );

describe("shared Workflow cards", () => {
  it("defaults to the latest version across pages, pins an older choice on refresh and keeps exact links and contracts together", async () => {
    const { sent } = renderApplication("/catalog/workflows", {
      "GET /v1/workflows": ({ url }) =>
        url.searchParams.get("cursor") === "next"
          ? json(page([workflow("10")]))
          : json(page([workflow("9")], "next")),
    });
    const card = await screen.findByRole("article", { name: "Source check" });
    expect(
      sent("GET", "/v1/workflows").some(
        (request) => new URL(request.url).searchParams.get("cursor") === "next",
      ),
    ).toBe(true);
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
    // A version without a presentation simply omits the purpose line.
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
