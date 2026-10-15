import { Link, useLocation } from "react-router";

import { TERMS } from "../../app/vocabulary";
import { ButtonLink } from "../../ui";
import { LIBRARY_SECTIONS, librarySectionOf } from "./navigation";
import "./library.css";

/**
 * The section tabs, shared by the Library pages and the Files page so both
 * show the same tabs. Styles live in library.css, imported here so every
 * page that shows the tabs loads them. They wrap onto further lines on narrow screens, so every
 * tab stays visible without scrolling.
 */
export function LibraryTabs() {
  const current = librarySectionOf(useLocation().pathname)?.to;
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
      <ButtonLink to="/catalog/studio">Open Node Studio</ButtonLink>
    </header>
  );
}
