/**
 * Project sections (contract §3 vocabulary). The URL segments stay as they
 * were (S06:1374-1375), only the names follow V3B. Runs and Workflows stay
 * one click away under "Advanced".
 */
export const MAIN_SECTIONS = [
  ["", "Overview"],
  ["artifacts", "Materials"],
  ["audits", "Checks"],
  ["findings", "Issues"],
  ["settings", "Settings"],
] as const;

export const ADVANCED_SECTIONS = [
  ["runs", "Runs"],
  ["workflows", "Workflows"],
] as const;

export const PROJECT_SECTIONS = [...MAIN_SECTIONS, ...ADVANCED_SECTIONS];

/** A section's URL segment; "" is the Overview. */
export type ProjectSectionSegment = (typeof PROJECT_SECTIONS)[number][0];

/** The URL of a project or one of its sections. */
export function projectPath(
  projectId: string,
  segment: ProjectSectionSegment = "",
): string {
  const root = `/projects/${encodeURIComponent(projectId)}`;
  return segment === "" ? root : `${root}/${segment}`;
}

/**
 * Start a check for this project (contract §5): `objective` prefills what to
 * check, `type` a check type by its profile name.
 */
export function startCheckPath(
  projectId: string,
  objective = "",
  type?: string,
): string {
  const params = new URLSearchParams({ project: projectId });
  const text = objective.trim();
  if (text !== "") params.set("objective", text);
  if (type !== undefined) params.set("type", type);
  return `/checks/new?${params.toString()}`;
}

/** The section a path inside /projects/:projectId shows. */
export function projectSectionOf(
  pathname: string,
  projectId: string,
): ProjectSectionSegment {
  const root = projectPath(projectId);
  return (
    PROJECT_SECTIONS.find(
      ([segment]) =>
        segment !== "" &&
        (pathname === `${root}/${segment}` ||
          pathname.startsWith(`${root}/${segment}/`)),
    )?.[0] ?? ""
  );
}
