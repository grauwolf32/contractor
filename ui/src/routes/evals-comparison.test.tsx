import { MemoryStorage, storedValues } from "../test/storage";
import { webcrypto } from "node:crypto";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import {
  createEvalFixture,
  EVAL_API_VERSION,
  EVAL_FIXTURE_ORIGIN,
} from "../test/evals-fixture";
import pairs from "../../../api/testdata/evals/valid/pair-page.json";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  localStorage.clear();
  vi.stubGlobal("crypto", webcrypto);
});
function start(
  fixture: ReturnType<typeof createEvalFixture>,
  path = "/evals/experiments/experiment-1/comparison",
) {
  const api = new PublicAPI(
    {
      uiVersion: "0.4.0",
      apiBaseUrl: EVAL_FIXTURE_ORIGIN,
      supportedApiVersions: [EVAL_API_VERSION],
    },
    async (input) =>
      fixture.fetch(input instanceof Request ? input : new Request(input)),
  );
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  return {
    ...render(<Application api={api} publicAPI={api} router={router} />),
    router,
  };
}

describe("Managed Eval comparisons", () => {
  it("uses complete server cohorts for charts and keeps missing usage distinct from zero", async () => {
    const fixture = createEvalFixture({ prepared: true }),
      user = userEvent.setup();
    start(fixture);
    const heading = await screen.findByRole("heading", {
      name: "Token distribution",
    });
    await waitFor(() =>
      expect(
        within(heading.closest("section")!).getByRole("cell", { name: "125" }),
      ).toBeVisible(),
    );
    expect(
      within(heading.closest("section")!).getByRole("cell", { name: "130" }),
    ).toBeVisible();
    await user.selectOptions(screen.getByLabelText("Pair filter"), "all");
    await screen.findByText(/4 matching pairs/);
    expect(screen.getAllByText(/Unavailable/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Quality regression/).length).toBeGreaterThan(0);
    expect(fixture.state.requests.some((r) => r.path === "/v1/runs")).toBe(
      false,
    );
    expect(screen.getByText(/Only complete matching pairs/)).toBeVisible();
  });

  it("keeps pair pagination/filter context on a round trip through evidence", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.secondPairPage = true;
    const user = userEvent.setup(),
      view = start(
        fixture,
        "/evals/experiments/experiment-1/comparison?filter=all&metric=duration",
      );
    await user.click(await screen.findByRole("button", { name: "Next pairs" }));
    await waitFor(() =>
      expect(view.router.state.location.search).toContain("cursor=pair-page-2"),
    );
    await user.click(
      screen.getAllByRole("link", { name: /unsafe-query \/ sample 1/ })[0]!,
    );
    await user.click(await screen.findByRole("link", { name: "← Comparison" }));
    expect(view.router.state.location.search).toContain("cursor=pair-page-2");
    expect(view.router.state.location.search).toContain("metric=duration");
    expect(view.router.state.location.search).toContain("filter=all");
  });

  it("drills down by a signed bin token and shows the same accessible bin counts", async () => {
    const fixture = createEvalFixture({ prepared: true }),
      user = userEvent.setup(),
      view = start(fixture);
    const bin = (
      await screen.findAllByRole("button", { name: /Open matching pairs/ })
    )[0]!;
    await user.click(bin);
    await waitFor(() =>
      expect(view.router.state.location.search).toContain("binFilter="),
    );
    expect(view.router.state.location.search).toContain("filter=all");
    expect(
      await screen.findByText(/Filtered to the selected distribution bin/),
    ).toBeVisible();
    await user.click(screen.getByText("Show bin data table"));
    expect(screen.getByRole("table", { name: "Shared bins" })).toBeVisible();
  });

  it("reuses a lost immutable review receipt and then selects with the reviewed CAS", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.lostAssessment = true;
    const user = userEvent.setup();
    start(
      fixture,
      `/evals/experiments/experiment-1/pairs/${pairs.items[0]!.pairId}`,
    );
    await user.click(await screen.findByRole("button", { name: "Review A" }));
    await screen.findByText("PRIVATE_RUBRIC_SENTINEL");
    await user.selectOptions(
      screen.getByLabelText("Decision for evidence-review"),
      "pass",
    );
    await user.type(
      screen.getByLabelText("Reason for evidence-review"),
      "Verified retained evidence",
    );
    await user.click(
      screen.getByRole("button", { name: "Save and select assessment" }),
    );
    await screen.findByText("Public API is unavailable");
    const retained = storedValues(localStorage);
    expect(retained).not.toContain("PRIVATE_");
    expect(retained).not.toContain("Verified retained evidence");
    await user.click(
      screen.getByRole("button", { name: "Save and select assessment" }),
    );
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", { name: "Review result" }),
      ).not.toBeInTheDocument(),
    );
    const submissions = fixture.state.requests.filter((r) =>
      r.path.endsWith("/assessments"),
    );
    expect(submissions).toHaveLength(2);
    expect(submissions[1]).toMatchObject({
      key: submissions[0]!.key,
      body: submissions[0]!.body,
    });
    expect(
      fixture.state.requests.find((r) => r.path.endsWith("/selections"))?.etag,
    ).toBe('"1"');
  });

  it("prevents a human verdict from applying to evidence changed since the pair snapshot", async () => {
    const fixture = createEvalFixture({ prepared: true });
    fixture.state.staleReview = true;
    const user = userEvent.setup();
    start(
      fixture,
      `/evals/experiments/experiment-1/pairs/${pairs.items[0]!.pairId}`,
    );
    await user.click(await screen.findByRole("button", { name: "Review A" }));
    await screen.findByText(/A newer result is selected/);
    expect(
      screen.getByRole("button", { name: "Save and select assessment" }),
    ).toBeDisabled();
    expect(
      fixture.state.requests.some((r) => r.path.endsWith("/assessments")),
    ).toBe(false);
  });

  it("uses the Audit execution workspace for diagnostics without counting child Runs as samples", async () => {
    const fixture = createEvalFixture({ prepared: true, audit: true });
    start(
      fixture,
      `/evals/experiments/experiment-1/pairs/${pairs.items[0]!.pairId}`,
    );
    const links = await screen.findAllByRole("link", { name: "audit audit-1" });
    expect(links[0]).toHaveAttribute(
      "href",
      "/projects/member-project/audits/audit-1",
    );
    expect(
      screen.getAllByText(
        /child Runs are evidence, not additional evaluation samples/,
      ),
    ).toHaveLength(2);
  });
});
