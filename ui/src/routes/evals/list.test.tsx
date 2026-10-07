import { webcrypto } from "node:crypto";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PublicAPI } from "../../api/client";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import {
  createEvalFixture,
  EVAL_API_VERSION,
  EVAL_FIXTURE_ORIGIN,
} from "../../test/evals-fixture";
import { MemoryStorage } from "../../test/storage";

beforeEach(() => {
  vi.stubGlobal("localStorage", new MemoryStorage());
  vi.stubGlobal("crypto", webcrypto);
});

function start(fixture: ReturnType<typeof createEvalFixture>, path: string) {
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
  render(<Application api={api} publicAPI={api} router={router} />);
  return router;
}

function preview() {
  return screen.getByRole("region", { name: "Experiment" });
}

describe("Experiments list", () => {
  it("keeps the selection in the URL and previews the selected experiment", async () => {
    const fixture = createEvalFixture({ prepared: true });
    const user = userEvent.setup();
    const router = start(fixture, "/evals?state=ready");
    const list = await screen.findByRole("region", { name: "Experiments" });
    expect(
      within(list).getByRole("heading", { level: 1, name: "Experiments" }),
    ).toBeVisible();
    expect(
      within(preview()).getByText("Choose an experiment"),
    ).toBeInTheDocument();

    const row = await within(list).findByRole("link", {
      name: "Trace instructions",
    });
    expect(row).toHaveAttribute(
      "href",
      "/evals?state=ready&experiment=experiment-1",
    );
    expect(
      within(list).getByText("Ready to start", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(within(list).getByText("Inconclusive")).toBeInTheDocument();
    await user.click(row);
    await waitFor(() =>
      expect(router.state.location.search).toBe(
        "?state=ready&experiment=experiment-1",
      ),
    );
    expect(row).toHaveAttribute("aria-current", "true");

    const detail = preview();
    expect(
      await within(detail).findByRole("heading", {
        name: "Trace instructions",
      }),
    ).toBeVisible();
    expect(
      within(detail).getByRole("link", { name: "Open experiment" }),
    ).toHaveAttribute("href", "/evals/experiments/experiment-1/overview");
    expect(
      within(detail).getByRole("button", { name: "Copy experiment ID" }),
    ).toBeInTheDocument();
    expect(
      await within(detail).findByText(
        "2 cases × 2 variants × 2 repetitions = 8 expected members",
      ),
    ).toBeInTheDocument();
    const variants = within(detail).getByRole("region", { name: "Variants" });
    expect(within(variants).getByText("A · Baseline")).toBeInTheDocument();
    expect(within(variants).getByTitle("trace-a@1")).toBeInTheDocument();
    expect(within(variants).getByText("B · Candidate")).toBeInTheDocument();
    expect(within(variants).getByTitle("trace-b@1")).toBeInTheDocument();
    const coverage = within(detail).getByRole("region", {
      name: "Experiment coverage",
    });
    expect(within(coverage).getByText("Inconclusive")).toBeInTheDocument();
    const sections = within(detail).getByRole("navigation", {
      name: "Open an experiment section",
    });
    expect(
      within(sections)
        .getAllByRole("link")
        .map((link) => link.getAttribute("href")),
    ).toEqual([
      "/evals/experiments/experiment-1/overview",
      "/evals/experiments/experiment-1/comparison",
      "/evals/experiments/experiment-1/attempts",
      "/evals/experiments/experiment-1/setup",
    ]);
  });

  it("moves the selection with J and opens it with Enter", async () => {
    const fixture = createEvalFixture({ prepared: true });
    const user = userEvent.setup();
    const router = start(fixture, "/evals");
    await screen.findByRole("link", { name: "Trace instructions" });
    await user.keyboard("j");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?experiment=experiment-1"),
    );
    await screen.findByRole("link", { name: "Open experiment" });
    const row = screen.getByRole("link", { name: "Trace instructions" });
    row.focus();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe(
        "/evals/experiments/experiment-1/overview",
      ),
    );
  });

  it("continues a draft's setup and labels the lifecycle filter in words", async () => {
    const fixture = createEvalFixture();
    const user = userEvent.setup();
    const router = start(fixture, "/evals?experiment=experiment-1");
    const detail = await screen.findByRole("region", { name: "Experiment" });
    expect(
      await within(detail).findByRole("link", { name: "Continue setup" }),
    ).toHaveAttribute("href", "/evals/experiments/experiment-1/setup");
    expect(
      within(preview()).getByText(
        "Configure the variants and cases, then prepare the experiment.",
      ),
    ).toBeInTheDocument();
    await user.click(screen.getByText("Filter experiments"));
    const lifecycle = screen.getByLabelText("Lifecycle");
    expect(
      within(lifecycle)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual([
      "All states",
      "Draft",
      "Preparing",
      "Ready to start",
      "Running",
      "Finishing",
      "Finished",
      "Pausing",
      "Paused",
      "Cancelling",
      "Cancelled",
    ]);
    await user.selectOptions(lifecycle, "draft");
    await waitFor(() =>
      expect(router.state.location.search).toContain("state=draft"),
    );
    await user.click(screen.getByRole("button", { name: "Clear filters" }));
    await waitFor(() =>
      expect(router.state.location.search).toBe("?experiment=experiment-1"),
    );
  });
});
