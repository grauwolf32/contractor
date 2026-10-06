/**
 * The rail destinations of the V3B shell (docs/design/ui/v3b-build-contract.md
 * §6) and the rules that decide which one is active. The rail and the
 * command palette's "Go to" group both read this list.
 */
import type { IconName } from "./icon";
import { capitalize, TERMS } from "./vocabulary";

export type DestinationId =
  | "inbox"
  | "projects"
  | "checks"
  | "issues"
  | "reports"
  | "runs"
  | "library"
  | "evals"
  | "operations";

export interface Destination {
  readonly id: DestinationId;
  readonly label: string;
  readonly to: string;
  readonly icon: IconName;
  /** Main group, the group under the divider, or the bottom of the rail. */
  readonly group: "main" | "more" | "bottom";
  /** Further words the command palette matches, such as former names. */
  readonly keywords: string;
  /** The session capability the destination needs. */
  readonly capability?: string;
}

export const DESTINATIONS: readonly Destination[] = [
  {
    id: "inbox",
    label: TERMS.inbox,
    to: "/",
    icon: "inbox",
    group: "main",
    keywords: "home decisions",
  },
  {
    id: "projects",
    label: "Projects",
    to: "/projects",
    icon: "projects",
    group: "main",
    keywords: "materials",
  },
  {
    id: "checks",
    label: capitalize(TERMS.checks),
    to: "/checks",
    icon: "checks",
    group: "main",
    keywords: "audits",
  },
  {
    id: "issues",
    label: capitalize(TERMS.issues),
    to: "/issues",
    icon: "issues",
    group: "main",
    keywords: "findings possible issues",
  },
  {
    id: "reports",
    label: capitalize(TERMS.reports),
    to: "/reports",
    icon: "reports",
    group: "main",
    keywords: "",
  },
  {
    id: "runs",
    label: "Runs",
    to: "/runs",
    icon: "runs",
    group: "more",
    keywords: "queue workflow runs",
  },
  {
    id: "library",
    label: TERMS.library,
    to: "/catalog",
    icon: "library",
    group: "more",
    keywords: "catalog check types workflows agents skills files artifacts",
  },
  {
    id: "evals",
    label: "Evals",
    to: "/evals",
    icon: "evals",
    group: "more",
    keywords: "evaluations experiments datasets",
  },
  {
    id: "operations",
    label: "Operations",
    to: "/operations",
    icon: "operations",
    group: "bottom",
    keywords: "runtime agents allocations configuration credentials",
    capability: "operations",
  },
];

/** The destinations a principal with these capabilities can open. */
export function destinationsFor(
  capabilities: readonly string[] | undefined,
): Destination[] {
  return DESTINATIONS.filter(
    (destination) =>
      destination.capability === undefined ||
      (capabilities?.includes(destination.capability) ?? false),
  );
}

function segmentsOf(pathname: string): string[] {
  return pathname.split("/").filter((segment) => segment !== "");
}

/**
 * The rail item a path belongs to: Inbox only on "/"; Checks on /checks… and
 * on a check page (/projects/:id/audits/:auditId…); Projects on the other
 * /projects… pages; Library on /catalog… and /artifacts…; Issues, Reports,
 * Runs, Evals and Operations on their own prefixes.
 */
export function activeDestination(pathname: string): DestinationId | undefined {
  const [first, , section, item] = segmentsOf(pathname);
  switch (first) {
    case undefined:
      return "inbox";
    case "checks":
      return "checks";
    case "projects":
      return section === "audits" && item !== undefined ? "checks" : "projects";
    case "issues":
      return "issues";
    case "reports":
      return "reports";
    case "runs":
      return "runs";
    case "catalog":
    case "artifacts":
      return "library";
    case "evals":
      return "evals";
    case "operations":
      return "operations";
    default:
      return undefined;
  }
}

/** The project a path is inside (/projects/:projectId…), if any. */
export function projectIdOf(pathname: string): string | undefined {
  const [first, projectId] = segmentsOf(pathname);
  if (first !== "projects" || projectId === undefined) return undefined;
  try {
    return decodeURIComponent(projectId);
  } catch {
    return undefined;
  }
}
