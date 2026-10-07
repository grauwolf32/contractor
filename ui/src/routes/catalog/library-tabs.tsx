import { NavLink } from "react-router";

import { TERMS } from "../../app/vocabulary";
import "./library.css";

/**
 * Library sections (contract §6): Check types, Workflows, Agents, Skills and
 * the personal Files library, which lives at /artifacts. Shared by the
 * Library pages and the Files page so both show the same tabs. Styles live in
 * library.css, imported here so every page that shows the tabs loads them.
 */
const LIBRARY_SECTIONS = [
  { to: "/catalog/audit-presets", label: "Check types" },
  { to: "/catalog/workflows", label: "Workflows" },
  { to: "/catalog/agents", label: "Agents" },
  { to: "/catalog/skills", label: "Skills" },
  { to: "/artifacts", label: "Files" },
] as const;

export function LibraryTabs({ className }: { className?: string }) {
  return (
    <nav
      className={["library-tabs", className].filter(Boolean).join(" ")}
      aria-label="Library sections"
    >
      {LIBRARY_SECTIONS.map((section) => (
        <NavLink key={section.to} to={section.to} className="library-tab">
          {section.label}
        </NavLink>
      ))}
    </nav>
  );
}

/**
 * The Library title with its section tabs: the top of every Library page.
 * The Files page (/artifacts) can show the same header.
 */
export function LibraryHeader() {
  return (
    <header className="library-header">
      <h1 className="library-title">{TERMS.library}</h1>
      <LibraryTabs />
    </header>
  );
}
