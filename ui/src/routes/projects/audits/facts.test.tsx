import { screen, within, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { projectFactsFixture } from "../../../test/project-facts-fixtures";
import { renderWithAPI } from "../../../test/render";
import { ProjectFindingsRoute } from "./findings";

function setup(facts = true) {
  const data = projectFactsFixture();
  const view = renderWithAPI(null, {
    path: "/projects/project_facts/facts",
    server: (request) => data.respond(request),
    routes: [
      {
        path: "/projects/:projectId/facts",
        element: <ProjectFindingsRoute facts={facts} />,
      },
    ],
  });
  return { data, ...view };
}

describe("Project Facts over canonical Findings", () => {
  it("bounds each load over 10,000 records and keeps one card for both receipts", async () => {
    const { data, user } = setup();
    const list = await screen.findByRole("list", { name: "Facts" });
    expect(within(list).getAllByRole("link")).toHaveLength(1);
    expect(
      screen.getByText("1 facts shown · 250 records loaded"),
    ).toBeVisible();
    expect(data.listReads).toEqual([0, 50, 100, 150, 200]);
    expect(data.payloadReads).toEqual([]);
    await user.click(screen.getByRole("button", { name: "Load more facts" }));
    await waitFor(() =>
      expect(within(list).getAllByRole("link")).toHaveLength(2),
    );
    expect(data.listReads).toEqual([
      0, 50, 100, 150, 200, 250, 300, 350, 400, 450,
    ]);
    expect(screen.getByLabelText("Severity")).toBeDisabled();
    expect(screen.queryByText("Unknown severity record")).toBeNull();
    const link = within(list).getByRole("link", {
      name: /Reported nginx deployment/,
    });
    expect(link).toHaveAttribute(
      "href",
      expect.stringContaining("/issues/project_facts/fact_nginx?state=all"),
    );
  });
  it("uses canonical analyst severity on explicit refresh without replacing identity", async () => {
    const { data, user } = setup();
    await screen.findByRole("list", { name: "Facts" });
    data.assess();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(
      await screen.findByText("No facts in the loaded records"),
    ).toBeVisible();
    expect(
      screen.queryByRole("link", { name: /Reported nginx deployment/ }),
    ).toBeNull();
    expect(data.fact.findingId).toBe("fact_nginx");
  });
  it("retains low and unknown severity in needs review and excludes informational suggestions", async () => {
    const { user } = setup(false);
    await screen.findByRole("list", { name: "Possible issues" });
    await user.selectOptions(screen.getByLabelText("Decision"), "needs_review");
    const list = screen.getByRole("list", { name: "Possible issues" });
    expect(
      within(list).getByRole("link", { name: /Unknown severity record/ }),
    ).toBeVisible();
    expect(
      within(list).getByRole("link", { name: /Low severity record/ }),
    ).toBeVisible();
    expect(
      within(list).queryByRole("link", { name: /Reported nginx deployment/ }),
    ).toBeNull();
  });
});
