export { catalogReturnState, locationDestination } from "../../app/navigation";
export type { CatalogReturnState } from "../../app/navigation";

interface LibrarySection {
  to: string;
  label: string;
  /** Further path prefixes that belong to the section. */
  also?: readonly string[];
}

/**
 * Library sections (contract §6): Check types, Workflows, Agents, Skills and
 * the personal Files library, which lives at /artifacts. Skill packages open
 * at /artifacts/skills/…, which belongs to Skills, not Files; Skills comes
 * first so it claims those paths.
 */
export const LIBRARY_SECTIONS: readonly LibrarySection[] = [
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
export function librarySectionOf(pathname: string): LibrarySection | undefined {
  return LIBRARY_SECTIONS.find((section) =>
    [section.to, ...(section.also ?? [])].some((prefix) =>
      under(pathname, prefix),
    ),
  );
}
