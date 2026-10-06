import { useContext, useEffect, useId, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { NavLink, useLocation, useNavigate } from "react-router";

import {
  ADVANCED_SECTIONS,
  MAIN_SECTIONS,
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
  const { pathname } = location;
  const navigate = useNavigate();
  const advancedLabel = useId();
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
  // the browser's own scroll position.
  const shownPath = useRef(pathname);
  useEffect(() => {
    if (shownPath.current === pathname) return;
    shownPath.current = pathname;
    const pane = bar.current?.closest<HTMLElement>(".ui-panes-detail");
    if (pane !== null && pane !== undefined) pane.scrollTop = 0;
    document
      .getElementById("main-content")
      ?.scrollIntoView?.({ block: "start" });
  }, [pathname]);

  function tab([segment, label]: readonly [string, string]) {
    return (
      <li key={segment}>
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

  // Wide panes show the tabs; narrow ones (phones, and the detail pane next
  // to the list on small laptops) show the labelled select instead.
  return (
    <div className="projects-tabs-bar" ref={bar}>
      <nav className="projects-tabs" aria-label="Project sections">
        <ul role="list" className="projects-tabs-list">
          {MAIN_SECTIONS.map(tab)}
        </ul>
        <div
          className="projects-tabs-advanced"
          role="group"
          aria-labelledby={advancedLabel}
        >
          <span className="projects-tabs-label" id={advancedLabel}>
            Advanced
          </span>
          <ul role="list" className="projects-tabs-list">
            {ADVANCED_SECTIONS.map(tab)}
          </ul>
        </div>
      </nav>
      <label className="projects-section-picker">
        <span>Project section</span>
        <select
          aria-label="Project section"
          value={current}
          onChange={(event) =>
            void navigate(destination(event.target.value), {
              state: navigationState,
            })
          }
        >
          {MAIN_SECTIONS.map(([segment, label]) => (
            <option key={segment} value={segment}>
              {label}
            </option>
          ))}
          <optgroup label="Advanced">
            {ADVANCED_SECTIONS.map(([segment, label]) => (
              <option key={segment} value={segment}>
                {label}
              </option>
            ))}
          </optgroup>
        </select>
      </label>
      <div className="projects-section-actions" ref={actionsRef} />
    </div>
  );
}
