import { type ReactNode, useEffect } from "react";
import { Navigate, NavLink, useLocation, useNavigate } from "react-router";

const sections = [
  ["", "Overview"],
  ["artifacts", "Artifacts"],
  ["workflows", "Workflows"],
  ["runs", "Runs"],
  ["audits", "Audits"],
  ["findings", "Findings"],
  ["settings", "Settings"],
] as const;

export function ProjectNavigation({ projectId }: { projectId: string }) {
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
      <label className="project-section-picker">
        Project section
        <select
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

export function ProjectLegacySectionRedirect({
  projectId,
  children,
}: {
  projectId: string;
  children: ReactNode;
}) {
  const location = useLocation();
  const root = `/projects/${encodeURIComponent(projectId)}`;
  const legacy: Record<string, string> = {
    "#project-overview": "settings",
    "#project-artifacts": "artifacts",
    "#project-workflows": "workflows",
    "#project-runs": "runs",
    "#project-audits": "audits",
  };
  const target = legacy[location.hash];
  if (location.pathname === root && target)
    return (
      <Navigate
        replace
        to={{ pathname: `${root}/${target}`, search: location.search }}
        state={location.state}
      />
    );
  return children;
}
