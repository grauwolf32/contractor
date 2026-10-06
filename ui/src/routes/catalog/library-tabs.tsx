import { NavLink } from "react-router";

/**
 * Library sections (contract §6): Check types, Workflows, Agents, Skills and
 * the personal Files library, which lives at /artifacts. Shared by the
 * Library pages and the Files page so both show the same tabs.
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
      className={["library-tabs", "section-navigation", className]
        .filter(Boolean)
        .join(" ")}
      aria-label="Library sections"
    >
      {LIBRARY_SECTIONS.map((section) => (
        <NavLink key={section.to} to={section.to}>
          {section.label}
        </NavLink>
      ))}
    </nav>
  );
}
