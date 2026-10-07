import { render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { describe, expect, it } from "vitest";

import { LibraryTabs } from "./library-tabs";

function renderTabsAt(path: string) {
  const router = createMemoryRouter([{ path: "*", element: <LibraryTabs /> }], {
    initialEntries: [path],
  });
  render(<RouterProvider router={router} />);
  return within(screen.getByRole("navigation", { name: "Library sections" }));
}

describe("Library tabs", () => {
  it("lists every section in order", () => {
    const tabs = renderTabsAt("/catalog/workflows");
    expect(
      tabs
        .getAllByRole("link")
        .map((link) => [link.textContent, link.getAttribute("href")]),
    ).toEqual([
      ["Check types", "/catalog/audit-presets"],
      ["Workflows", "/catalog/workflows"],
      ["Agents", "/catalog/agents"],
      ["Skills", "/catalog/skills"],
      ["Files", "/artifacts"],
    ]);
  });

  it.each([
    ["/catalog/audit-presets", "Check types"],
    ["/catalog/audit-presets/owasp-top10-2025-source-risk/1", "Check types"],
    ["/catalog/workflows/openapi-from-source/1", "Workflows"],
    ["/catalog/agents/source-reviewer/2", "Agents"],
    ["/catalog/skills", "Skills"],
    // Skill packages open as files but belong to Skills.
    ["/artifacts/skills/openapi-analysis", "Skills"],
    ["/artifacts", "Files"],
    ["/artifacts/inputs/wordlist", "Files"],
    // A namespace that only starts like "skills" is an ordinary file.
    ["/artifacts/skills-archive/notes", "Files"],
  ])("marks only the section of %s as current: %s", (path, current) => {
    const tabs = renderTabsAt(path);
    expect(
      tabs
        .getAllByRole("link")
        .filter((link) => link.getAttribute("aria-current") === "page")
        .map((link) => link.textContent),
    ).toEqual([current]);
  });

  it("marks no section outside the Library", () => {
    const tabs = renderTabsAt("/projects/project_a/artifacts");
    expect(
      tabs
        .getAllByRole("link")
        .filter((link) => link.hasAttribute("aria-current")),
    ).toEqual([]);
  });
});
