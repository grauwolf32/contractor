import { Navigate, Outlet, useLocation } from "react-router";

import { useDocumentTitle } from "../../app/document-title";
import { TERMS } from "../../app/vocabulary";
import { LibraryHeader } from "./library-tabs";

import "./catalog.css";

/** /catalog opens the first Library section, keeping its query and hash. */
export function CatalogIndexRedirect() {
  const { search, hash } = useLocation();
  return (
    <Navigate
      replace
      to={{ pathname: "/catalog/audit-presets", search, hash }}
    />
  );
}

const LIBRARY_SECTIONS = [
  { prefix: "/catalog/audit-presets", label: "Check types" },
  { prefix: "/catalog/workflows", label: "Workflows" },
  { prefix: "/catalog/agents", label: "Agents" },
  { prefix: "/catalog/skills", label: "Skills" },
] as const;

/** The section a Library path belongs to, for the document title. */
function librarySectionLabel(pathname: string): string | undefined {
  return LIBRARY_SECTIONS.find(
    (section) =>
      pathname === section.prefix || pathname.startsWith(`${section.prefix}/`),
  )?.label;
}

/** Library: the title and section tabs above every section page. */
export function CatalogLayoutRoute() {
  const { pathname } = useLocation();
  const section = librarySectionLabel(pathname);
  useDocumentTitle(
    section === undefined ? TERMS.library : `${section} · ${TERMS.library}`,
  );
  return (
    <div className="library-page">
      <LibraryHeader />
      <Outlet />
    </div>
  );
}
