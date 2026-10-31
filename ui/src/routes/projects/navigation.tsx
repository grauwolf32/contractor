import { useContext, useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { NavLink, useLocation } from "react-router";

import {
  PROJECT_SECTIONS,
  projectPath,
  projectSectionOf,
} from "./project-sections";
import { ProjectSectionActionsContext } from "./section-actions-context";

/** Slim toolbar for a project section: rendered on the tab-bar row. */
export function ProjectSectionActions({ children }: { children: ReactNode }) {
  const slot = useContext(ProjectSectionActionsContext);
  if (slot === undefined)
    return <div className="project-section-actions">{children}</div>;
  return slot === null ? null : createPortal(children, slot);
}

export function ProjectNavigation({
  projectId,
  actionsRef,
}: {
  projectId: string;
  actionsRef?: (element: HTMLDivElement | null) => void;
}) {
  const root = projectPath(projectId);
  const location = useLocation();
  const { pathname, hash } = location;
  const bar = useRef<HTMLDivElement>(null);
  const current = projectSectionOf(pathname, projectId);
  // Each section keeps its own query (filters, cursors, versions) in the
  // history state, so switching sections and back restores it.
  const saved: unknown = location.state?.projectSections;
  const sectionQueries: Record<string, string> = Object.fromEntries(
    PROJECT_SECTIONS.map(([segment]) => {
      const value =
        typeof saved === "object" && saved !== null
          ? (saved as Record<string, unknown>)[segment]
          : undefined;
      return [
        segment,
        segment === current
          ? location.search
          : typeof value === "string" && value.startsWith("?")
            ? value
            : "",
      ];
    }),
  );
  const navigationState = {
    ...location.state,
    projectSections: sectionQueries,
  };
  function destination(segment: string) {
    return `${root}${segment ? `/${segment}` : ""}${sectionQueries[segment] ?? ""}`;
  }
  // A section opened from another one starts at its top: in the detail pane
  // on wide screens, in the page on phones. A page that loads directly keeps
  // the browser's own scroll position, and a link to an anchor of a section
  // (settings#live-target) leaves scrolling to the section.
  const shownPath = useRef(pathname);
  useEffect(() => {
    if (shownPath.current === pathname) return;
    shownPath.current = pathname;
    if (hash !== "") return;
    const pane = bar.current?.closest<HTMLElement>(".ui-panes-detail");
    if (pane !== null && pane !== undefined) pane.scrollTop = 0;
    document
      .getElementById("main-content")
      ?.scrollIntoView?.({ block: "start" });
  }, [hash, pathname]);

  // Reveal only along the horizontal axis: scrollIntoView would also move
  // the page or its detail pane when opening a section or resizing it.
  useEffect(() => {
    const nav = bar.current?.querySelector("nav");
    if (!nav) return;
    function revealSelected() {
      const active = nav?.querySelector('[aria-current="page"]');
      if (!active || !nav) return;
      const viewport = nav.getBoundingClientRect();
      const item = active.getBoundingClientRect();
      if (item.left < viewport.left)
        nav.scrollLeft += item.left - viewport.left;
      else if (item.right > viewport.right)
        nav.scrollLeft += item.right - viewport.right;
    }
    revealSelected();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(revealSelected);
    observer.observe(nav);
    return () => observer.disconnect();
  }, [pathname]);

  function tab([segment, label]: readonly [string, string]) {
    return (
      <li
        key={segment}
        className={segment === "runs" ? "projects-tab-divider" : undefined}
      >
        <NavLink
          className="projects-tab"
          to={destination(segment)}
          state={navigationState}
          end={segment === ""}
        >
          {label}
        </NavLink>
      </li>
    );
  }

  return (
    <div className="projects-tabs-bar" ref={bar}>
      <nav className="projects-tabs" aria-label="Project sections">
        <ul role="list" className="projects-tabs-list">
          {PROJECT_SECTIONS.map(tab)}
        </ul>
      </nav>
      <div className="projects-section-actions" ref={actionsRef} />
    </div>
  );
}
