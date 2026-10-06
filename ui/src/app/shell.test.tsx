import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  auditFixture,
  projectFixture,
  renderShell,
  reviewFixture,
} from "../test/shell-harness";
import { activeDestination } from "./destinations";

function primaryNavigation() {
  return screen.getByRole("navigation", { name: "Primary navigation" });
}

const ACTIVE_ITEMS: readonly [path: string, item: string][] = [
  ["/", "Inbox"],
  ["/projects", "Projects"],
  ["/projects?new=1", "Projects"],
  ["/projects/project_a", "Projects"],
  ["/projects/project_a/artifacts", "Projects"],
  ["/projects/project_a/audits", "Projects"],
  ["/projects/project_a/findings", "Projects"],
  ["/projects/project_a/audits/audit_a", "Checks"],
  ["/projects/project_a/audits/audit_a/coverage", "Checks"],
  ["/checks", "Checks"],
  ["/checks/new", "Checks"],
  ["/issues", "Issues"],
  ["/issues/audit_a/finding_a", "Issues"],
  ["/reports", "Reports"],
  ["/reports/audit_a", "Reports"],
  ["/runs", "Runs"],
  ["/runs/run_a/artifacts/outputs/report", "Runs"],
  ["/catalog", "Library"],
  ["/catalog/workflows/review/2", "Library"],
  ["/artifacts", "Library"],
  ["/artifacts/skills/review", "Library"],
  ["/evals", "Evals"],
  ["/evals/experiments/experiment_a/comparison", "Evals"],
  ["/operations", "Operations"],
  ["/operations/settings", "Operations"],
];

describe("application shell rail", () => {
  it.each(ACTIVE_ITEMS)("marks %s as %s", async (path, item) => {
    renderShell(path);
    await screen.findByText(`Page at ${path}`);
    const current = within(primaryNavigation())
      .getAllByRole("link")
      .filter((link) => link.getAttribute("aria-current") === "page");
    expect(current.map((link) => link.textContent)).toEqual([item]);
  });

  it("marks nothing outside the rail destinations", () => {
    expect(activeDestination("/login")).toBeUndefined();
    expect(activeDestination("/checksum")).toBeUndefined();
    expect(activeDestination("/project")).toBeUndefined();
  });

  it("lists the destinations in three groups, Operations only for operators", async () => {
    const { unmount } = renderShell("/runs");
    await screen.findByText("Page at /runs");
    const lists = within(primaryNavigation()).getAllByRole("list");
    expect(
      lists.map((list) =>
        within(list)
          .getAllByRole("link")
          .map((link) => [link.textContent, link.getAttribute("href")]),
      ),
    ).toEqual([
      [
        ["Inbox", "/"],
        ["Projects", "/projects"],
        ["Checks", "/checks"],
        ["Issues", "/issues"],
        ["Reports", "/reports"],
      ],
      [
        ["Runs", "/runs"],
        ["Library", "/catalog"],
        ["Evals", "/evals"],
      ],
      [["Operations", "/operations"]],
    ]);
    unmount();

    renderShell("/operations/settings", { capabilities: ["user"] });
    await screen.findByText("Page at /operations/settings");
    expect(
      within(primaryNavigation()).queryByRole("link", { name: "Operations" }),
    ).toBeNull();
    expect(
      within(primaryNavigation()).queryByRole("link", { current: "page" }),
    ).toBeNull();
    expect(
      within(primaryNavigation()).getByRole("button", { name: "Account" }),
    ).toBeVisible();
  });

  it("keeps the skip link, one main region and one primary navigation", async () => {
    renderShell("/projects");
    await screen.findByText("Page at /projects");
    expect(
      screen.getByRole("link", { name: "Skip to content" }),
    ).toHaveAttribute("href", "#main-content");
    const main = screen.getByRole("main");
    expect(main).toHaveAttribute("id", "main-content");
    expect(main).toHaveAttribute("tabindex", "-1");
    expect(main).toHaveTextContent("Page at /projects");
    expect(
      screen.getAllByRole("navigation", { name: "Primary navigation" }),
    ).toHaveLength(1);
    expect(screen.getByRole("link", { name: "Contractor" })).toHaveAttribute(
      "href",
      "/",
    );
  });
});

describe("Inbox badge", () => {
  const server = (proposed: number, otherDecisions = 0) => ({
    projects: [projectFixture("project_a")],
    audits: {
      project_a: [
        auditFixture("project_a", "audit_running", "active"),
        auditFixture("project_a", "audit_done", "completed"),
      ],
    },
    proposedFindings: { audit_running: proposed, audit_done: 7 },
    reviews: {
      audit_running: Array.from({ length: otherDecisions }, (_, index) =>
        reviewFixture(
          "audit_running",
          `request_${index}`,
          "active-check-approval",
        ),
      ),
    },
  });

  it("counts what needs a decision and says so to screen readers", async () => {
    renderShell("/runs", { server: server(2, 1) });
    const inbox = await within(primaryNavigation()).findByRole("link", {
      name: "Inbox (3 need your decision)",
    });
    expect(inbox).toHaveAttribute("href", "/");
    const badge = inbox.querySelector(".shell-nav-badge");
    expect(badge).toHaveTextContent(/^3$/);
    expect(badge?.closest("[aria-hidden='true']")).not.toBeNull();
  });

  it("uses the singular for one decision", async () => {
    renderShell("/runs", { server: server(1) });
    expect(
      await within(primaryNavigation()).findByRole("link", {
        name: "Inbox (1 needs your decision)",
      }),
    ).toBeVisible();
  });

  it("caps the visible count at 99+", async () => {
    renderShell("/runs", { server: server(120) });
    const inbox = await within(primaryNavigation()).findByRole("link", {
      name: "Inbox (120 need your decision)",
    });
    expect(inbox.querySelector(".shell-nav-badge")).toHaveTextContent(/^99\+$/);
  });

  it("shows no badge when nothing waits", async () => {
    const { requests } = renderShell("/runs", { server: server(0) });
    await waitFor(() =>
      expect(requests.some((url) => url.pathname.endsWith("/reviews"))).toBe(
        true,
      ),
    );
    const inbox = within(primaryNavigation()).getByRole("link", {
      name: "Inbox",
    });
    expect(inbox.querySelector(".shell-nav-badge")).toBeNull();
  });
});

describe("phone menu", () => {
  it("toggles the navigation as a drawer with the account items", async () => {
    const user = userEvent.setup();
    renderShell("/runs");
    await screen.findByText("Page at /runs");
    const toggle = screen.getByRole("button", { name: "Menu" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAttribute("aria-controls", primaryNavigation().id);
    expect(
      within(primaryNavigation()).queryByRole("button", { name: "Sign out" }),
    ).toBeNull();

    const page = screen.getByRole("main").parentElement;
    expect(page).not.toHaveAttribute("inert");

    await user.click(toggle);
    expect(toggle).toHaveAccessibleName("Close menu");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    // The drawer covers the page, so the page leaves the focus order.
    expect(page).toHaveAttribute("inert");
    const drawer = primaryNavigation();
    expect(within(drawer).getByText("owner")).toBeVisible();
    expect(
      within(drawer).getByRole("link", { name: "Settings" }),
    ).toHaveAttribute("href", "/operations/settings");
    expect(within(drawer).getByRole("group", { name: "Theme" })).toBeVisible();
    expect(
      within(drawer).getByRole("button", { name: "Sign out" }),
    ).toBeVisible();
    expect(within(drawer).getByText(/^UI \S+$/)).toBeVisible();
    expect(
      screen.getAllByRole("navigation", { name: "Primary navigation" }),
    ).toHaveLength(1);

    within(drawer).getByRole("link", { name: "Projects" }).focus();
    await user.keyboard("{Escape}");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveAccessibleName("Menu");
    expect(toggle).toHaveFocus();
    expect(page).not.toHaveAttribute("inert");
  });

  it("closes the drawer after navigating", async () => {
    const user = userEvent.setup();
    const { router } = renderShell("/runs");
    await screen.findByText("Page at /runs");
    const toggle = screen.getByRole("button", { name: "Menu" });
    await user.click(toggle);
    await user.click(
      within(primaryNavigation()).getByRole("link", { name: "Issues" }),
    );
    expect(await screen.findByText("Page at /issues")).toBeVisible();
    expect(router.state.location.pathname).toBe("/issues");
    expect(toggle).toHaveAttribute("aria-expanded", "false");

    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await act(async () => {
      await router.navigate("/reports");
    });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });

  it("stays closed when history returns to where it was opened", async () => {
    const user = userEvent.setup();
    const { router } = renderShell("/issues");
    await act(async () => {
      await router.navigate("/runs");
    });
    const toggle = screen.getByRole("button", { name: "Menu" });
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");

    // Back and Forward restore the entry the drawer was opened at.
    await act(async () => {
      await router.navigate(-1);
    });
    expect(await screen.findByText("Page at /issues")).toBeVisible();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await act(async () => {
      await router.navigate(1);
    });
    expect(await screen.findByText("Page at /runs")).toBeVisible();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("main").parentElement).not.toHaveAttribute("inert");

    // A navigation that does not go through the drawer, then Back.
    await user.click(toggle);
    await act(async () => {
      await router.navigate("/reports");
    });
    await act(async () => {
      await router.navigate(-1);
    });
    expect(await screen.findByText("Page at /runs")).toBeVisible();
    expect(toggle).toHaveAttribute("aria-expanded", "false");
  });
});
