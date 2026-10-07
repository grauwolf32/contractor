import { Link, useLocation } from "react-router";

import { TERMS } from "../../app/vocabulary";
import "./library.css";

interface LibrarySection {
  to: string;
  label: string;
  /** Further path prefixes that belong to the section. */
  also?: readonly string[];
}

/**
 * Library sections (contract §6): Check types, Workflows, Agents, Skills and
 * the personal Files library, which lives at /artifacts. Shared by the
 * Library pages and the Files page so both show the same tabs. Skill
 * packages open at /artifacts/skills/…, which belongs to Skills, not Files;
 * Skills comes first so it claims those paths. Styles live in library.css,
 * imported here so every page that shows the tabs loads them.
 */
const LIBRARY_SECTIONS: readonly LibrarySection[] = [
  { to: "/catalog/audit-presets", label: "Check types" },
  { to: "/catalog/workflows", label: "Workflows" },
  { to: "/catalog/agents", label: "Agents" },
  { to: "/catalog/skills", label: "Skills", also: ["/artifacts/skills"] },
  { to: "/artifacts", label: "Files" },
];

function under(pathname: string, prefix: string): boolean {
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}

/** The Library section a path belongs to, if any. */
function librarySectionOf(pathname: string): string | undefined {
  return LIBRARY_SECTIONS.find((section) =>
    [section.to, ...(section.also ?? [])].some((prefix) =>
      under(pathname, prefix),
    ),
  )?.to;
}

/**
 * The section tabs. They wrap onto further lines on narrow screens, so every
 * tab stays visible without scrolling.
 */
export function LibraryTabs() {
  const current = librarySectionOf(useLocation().pathname);
  return (
    <nav className="library-tabs" aria-label="Library sections">
      {LIBRARY_SECTIONS.map((section) => (
        <Link
          key={section.to}
          to={section.to}
          className="library-tab"
          aria-current={section.to === current ? "page" : undefined}
        >
          {section.label}
        </Link>
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
      <Link className="ui-btn" to="/catalog/studio">
        Open Node Studio
      </Link>
    </header>
  );
}
