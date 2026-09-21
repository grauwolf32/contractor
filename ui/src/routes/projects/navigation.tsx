import { useContext, useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { NavLink, useLocation, useNavigate } from "react-router";
import { ProjectSectionActionsContext } from "./section-actions-context";

/** Slim toolbar for a project section: rendered on the tab-bar row. */
export function ProjectSectionActions({ children }: { children: ReactNode }) {
  const slot = useContext(ProjectSectionActionsContext);
  if (slot === undefined)
    return <div className="project-section-actions">{children}</div>;
  return slot === null ? null : createPortal(children, slot);
}

const sections = [
  ["", "Overview"],
  ["artifacts", "Artifacts"],
  ["workflows", "Workflows"],
  ["runs", "Runs"],
  ["audits", "Audits"],
  ["findings", "Findings"],
  ["settings", "Settings"],
] as const;

export function ProjectNavigation({
  projectId,
  actionsRef,
}: {
  projectId: string;
  actionsRef?: (element: HTMLDivElement | null) => void;
}) {
  const root = `/projects/${encodeURIComponent(projectId)}`;
  const location = useLocation();
  const { pathname } = location;
  const navigate = useNavigate();
  const current =
    sections.find(
      ([segment]) =>
        segment !== "" && pathname.startsWith(`${root}/${segment}`),
    )?.[0] ?? "";
  const saved = location.state?.projectSections;
  const sectionQueries: Record<string, string> = Object.fromEntries(
    sections.map(([segment]) => {
      const value = saved?.[segment];
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
  useEffect(() => {
    document
      .getElementById("main-content")
      ?.scrollIntoView?.({ block: "start" });
  }, [pathname]);
  return (
    <>
      <div className="project-section-bar">
        <nav
          className="project-section-navigation section-navigation"
          aria-label="Project sections"
        >
          {sections.map(([segment, label]) => (
            <NavLink
              key={segment}
              to={destination(segment)}
              state={navigationState}
              end={segment === ""}
            >
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="project-section-actions" ref={actionsRef} />
      </div>
      <label className="project-section-picker">
        Project section
        <select
          aria-label="Project section"
          value={current}
          onChange={(event) =>
            void navigate(destination(event.target.value), {
              state: navigationState,
            })
          }
        >
          {sections.map(([segment, label]) => (
            <option key={segment} value={segment}>
              {label}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}
