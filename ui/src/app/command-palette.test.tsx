import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import {
  newerPresetFixture,
  presetFixture,
} from "../test/audit-presets-fixture";
import { RESULTS_PER_GROUP } from "./command-palette";
import {
  auditFixture,
  type FakeServer,
  projectFixture,
  renderShell,
  workflowFixture,
} from "./shell-test-harness";

const server: FakeServer = {
  projects: [
    projectFixture("project_pay", { name: "Payments core" }),
    projectFixture("project_alpha", {
      name: "Alpha payments",
      description: "Card processing\nSecond line",
    }),
  ],
  audits: {
    project_pay: [
      auditFixture(
        "project_pay",
        "audit_idor",
        "active",
        "Find IDOR in orders",
      ),
    ],
  },
  profiles: [presetFixture, newerPresetFixture],
  workflows: [
    workflowFixture("openapi-from-source", "1"),
    workflowFixture("openapi-from-source", "2", "OpenAPI from source"),
  ],
};

async function openWithKeyboard() {
  const user = userEvent.setup();
  await user.keyboard("{Control>}k{/Control}");
  const dialog = await screen.findByRole("dialog", {
    name: "Search or start a check",
  });
  return { user, dialog, field: within(dialog).getByRole("combobox") };
}

function options() {
  return screen
    .queryAllByRole("option")
    .map((option) => option.querySelector(".shell-palette-label")?.textContent);
}

function group(name: string) {
  return screen.getByRole("group", { name });
}

describe("command palette", () => {
  it("opens with Ctrl+K and closes with it again, restoring focus", async () => {
    renderShell("/runs", { server });
    await screen.findByText("Page at /runs");
    const runs = screen.getByRole("link", { name: "Runs" });
    runs.focus();
    const { user, field } = await openWithKeyboard();
    expect(field).toHaveFocus();
    expect(field).toHaveAttribute("aria-expanded", "true");
    expect(field).toHaveAttribute(
      "aria-controls",
      screen.getByRole("listbox").id,
    );
    expect(
      within(group("Actions"))
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["Start a check", "New project", "Settings"]);
    expect(
      within(group("Go to"))
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual([
      "Inbox",
      "Projects",
      "Checks",
      "Issues",
      "Reports",
      "Runs",
      "Library",
      // The Library's personal files: no rail item, one jump away here.
      "FilesLibrary",
      "Evals",
      "Operations",
    ]);

    await user.keyboard("{Control>}k{/Control}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(runs).toHaveFocus());
  });

  it("opens from the top bar and closes with Escape, restoring focus", async () => {
    const user = userEvent.setup();
    renderShell("/runs", { server });
    const command = await screen.findByRole("button", {
      name: "Search or start a check…",
    });
    expect(command).toHaveAttribute("aria-keyshortcuts", "Control+K");
    await user.click(command);
    expect(
      await screen.findByRole("combobox", { name: "Search or start a check" }),
    ).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(command).toHaveFocus());
  });

  it("opens from the phone top bar and closes with its Close button", async () => {
    const user = userEvent.setup();
    renderShell("/runs", { server });
    const command = await screen.findByRole("button", {
      name: "Search or start a check",
    });
    await user.click(command);
    const dialog = await screen.findByRole("dialog");
    const close = within(dialog).getByRole("button", { name: "Close" });
    expect(close).toHaveAttribute("aria-keyshortcuts", "Escape");
    await user.click(close);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(command).toHaveFocus());
  });

  it("filters every group by all words, best matches first", async () => {
    renderShell("/runs", { server });
    const { user, field } = await openWithKeyboard();
    await user.type(field, "pay");
    expect(
      await within(group("Projects")).findAllByRole("option"),
    ).toHaveLength(2);
    // Label prefix first; the check matches through its project's name.
    expect(options()).toEqual([
      "Payments core",
      "Alpha payments",
      "Find IDOR in orders",
    ]);
    expect(screen.queryByRole("group", { name: "Go to" })).toBeNull();

    await user.clear(field);
    await user.type(field, "FIND orders");
    expect(
      await within(group("Checks")).findByRole("option"),
    ).toHaveTextContent("Find IDOR in orders");
    expect(options()).toEqual(["Find IDOR in orders"]);

    await user.clear(field);
    await user.type(field, "library");
    expect(options()).toEqual(["Library", "Files"]);

    await user.clear(field);
    await user.type(field, "artifacts");
    expect(options()).toEqual(["Library", "Files"]);

    await user.clear(field);
    await user.type(field, "catalog");
    expect(options()).toEqual(["Library"]);

    // Hidden keywords match at word starts only: "re" is not in "create".
    await user.clear(field);
    await user.type(field, "re");
    expect(options()).toEqual(["Reports", "Payments core", "source-review"]);
  });

  it("says when nothing matches", async () => {
    renderShell("/runs", { server });
    const { user, field } = await openWithKeyboard();
    await user.type(field, "zzzz");
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("No matches"),
    );
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(field).toHaveAttribute("aria-expanded", "false");
    expect(field).not.toHaveAttribute("aria-activedescendant");
  });

  it(`shows at most ${RESULTS_PER_GROUP} results per group`, async () => {
    renderShell("/runs", {
      server: {
        projects: Array.from({ length: 9 }, (_, index) =>
          projectFixture(`project_${index}`, { name: `Service ${index}` }),
        ),
      },
    });
    const { user, field } = await openWithKeyboard();
    await user.type(field, "service");
    await waitFor(() =>
      expect(within(group("Projects")).getAllByRole("option")).toHaveLength(
        RESULTS_PER_GROUP,
      ),
    );
  });

  it("moves with the arrow keys, wrapping, and opens with Enter", async () => {
    const { router } = renderShell("/runs", { server });
    const { user, field } = await openWithKeyboard();
    const selected = () =>
      screen
        .getAllByRole("option")
        .filter((option) => option.getAttribute("aria-selected") === "true");
    const [first] = selected();
    expect(first).toHaveTextContent("Start a check");
    expect(field).toHaveAttribute("aria-activedescendant", first?.id);

    await user.keyboard("{ArrowDown}");
    const [second] = selected();
    expect(second).toHaveTextContent("New project");
    expect(field).toHaveAttribute("aria-activedescendant", second?.id);
    expect(first).toHaveAttribute("aria-selected", "false");

    await user.keyboard("{ArrowUp}{ArrowUp}");
    expect(selected().map((option) => option.textContent)).toEqual([
      "Operations",
    ]);
    await user.keyboard("{ArrowUp}{ArrowUp}{ArrowUp}");
    expect(selected().map((option) => option.textContent)).toEqual(["Library"]);
    await user.keyboard("{Enter}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/catalog"),
    );
  });

  it("opens projects, checks, check types and workflows", async () => {
    const { router } = renderShell("/runs", { server });
    const cases: [query: string, label: string, path: string][] = [
      ["files", "Files", "/artifacts"],
      ["source-review", "source-review", "/checks/new?type=source-review"],
      [
        "openapi from",
        "OpenAPI from source",
        "/catalog/workflows/openapi-from-source/2",
      ],
      ["payments core", "Payments core", "/projects/project_pay"],
      // Inside the project now: its check, and check types keep the project.
      [
        "idor",
        "Find IDOR in orders",
        "/projects/project_pay/audits/audit_idor",
      ],
      [
        "source-review",
        "source-review",
        "/checks/new?project=project_pay&type=source-review",
      ],
    ];
    for (const [query, label, path] of cases) {
      const { user, field } = await openWithKeyboard();
      await user.type(field, query);
      const option = await screen.findByRole("option", {
        name: new RegExp(`^${label}`),
      });
      await user.click(option);
      await waitFor(() =>
        expect(
          `${router.state.location.pathname}${router.state.location.search}`,
        ).toBe(path),
      );
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    }
  });

  it("lists one entry per check type and workflow, at its newest version", async () => {
    renderShell("/runs", { server });
    const { user, field } = await openWithKeyboard();
    await user.type(field, "source");
    expect(
      await within(group("Check types")).findAllByRole("option"),
    ).toHaveLength(1);
    const workflow = within(group("Workflows")).getByRole("option");
    expect(workflow).toHaveTextContent("OpenAPI from source");
    expect(workflow).toHaveTextContent("openapi-from-source@2");
  });

  it("starts a check in the project the page belongs to", async () => {
    const { router } = renderShell("/projects/project_pay/audits", { server });
    const { user } = await openWithKeyboard();
    const start = within(group("Actions")).getByRole("option", {
      name: /^Start a check/,
    });
    expect(start).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(start).toHaveTextContent("In Payments core"));
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(router.state.location.search).toBe("?project=project_pay"),
    );
    expect(router.state.location.pathname).toBe("/checks/new");
  });

  it("starts a check without a project elsewhere", async () => {
    const { router } = renderShell("/issues", { server });
    const { user } = await openWithKeyboard();
    await user.keyboard("{Enter}");
    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/checks/new"),
    );
    expect(router.state.location.search).toBe("");
  });

  it("offers Operations only with the capability", async () => {
    renderShell("/runs", { server, capabilities: ["user"] });
    const { user, field } = await openWithKeyboard();
    expect(
      within(group("Go to")).queryByRole("option", { name: "Operations" }),
    ).toBeNull();
    await user.type(field, "operations");
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("No matches"),
    );
    expect(screen.queryByRole("option", { name: "Operations" })).toBeNull();
  });

  it("reads check types and workflows only while it is open", async () => {
    const { requests } = renderShell("/runs", { server });
    await screen.findByText("Page at /runs");
    await waitFor(() =>
      expect(requests.some((url) => url.pathname === "/v1/projects")).toBe(
        true,
      ),
    );
    const catalogReads = () =>
      requests.filter(
        (url) =>
          url.pathname === "/v1/audit-profiles" ||
          url.pathname === "/v1/workflows",
      );
    expect(catalogReads()).toEqual([]);
    await openWithKeyboard();
    await waitFor(() => expect(catalogReads()).toHaveLength(2));
  });
});
