import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  workflowResource,
  workflowSummary,
} from "../../test/builders-execution";
import { json, page } from "../../test/responses";
import { renderApplication } from "../../test/application";

const workflow = (version: string) =>
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
  });

describe("shared Workflow cards", () => {
  it("defaults to the latest version across pages, pins an older choice on refresh and keeps exact links and contracts together", async () => {
    const { sent } = renderApplication("/catalog/workflows", {
      server: {
        "GET /v1/workflows/{name}/versions/{version}": ({ params }) =>
          json(workflow(params.version!)),
        "GET /v1/workflows": ({ url }) =>
          url.searchParams.get("cursor") === "next"
            ? json(page([workflowSummary(workflow("10"))]))
            : json(page([workflowSummary(workflow("9"))], "next")),
      },
    });
    await screen.findByRole("heading", { name: "Source check" });
    const card = screen.getByRole("region", { name: "Workflows details" });
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
      screen.getByRole("link", { name: /Workflow (Source check|Check)/ }),
    ).toHaveAttribute("href", "/catalog/workflows/check/10");
    expect(within(card).getByText("Report")).toBeVisible();
    expect(
      within(card).getByText("Review source evidence."),
    ).toBeInTheDocument();
    const user = userEvent.setup();
    await user.selectOptions(select, "9");
    await screen.findByRole("heading", { name: "Check" });
    expect(within(card).queryByText("Report")).toBeNull();
    // A version without a presentation simply omits the purpose line.
    expect(within(card).queryByText("Review source evidence.")).toBeNull();
    expect(within(card).queryByText(/Purpose is not described/)).toBeNull();
    expect(card.querySelector(".workflow-card-description")).toBeNull();
    expect(
      screen.getByRole("link", { name: /Workflow (Source check|Check)/ }),
    ).toHaveAttribute("href", "/catalog/workflows/check/9");
    await user.click(within(card).getByRole("button", { name: "Refresh" }));
    await waitFor(() =>
      expect(
        within(card).getByRole("button", { name: "Refresh" }),
      ).toBeEnabled(),
    );
    expect(
      within(card).getByRole("combobox", { name: "Version of check" }),
    ).toHaveValue("9");
  });
});
