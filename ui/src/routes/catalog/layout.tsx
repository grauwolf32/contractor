import { Navigate, Outlet, useLocation } from "react-router";

import { useDocumentTitle } from "../../app/document-title";
import { TERMS } from "../../app/vocabulary";
import { LibraryHeader } from "./library-tabs";
import { librarySectionOf } from "./navigation";

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

/** Library: the title and section tabs above every section page. */
export function CatalogLayoutRoute() {
  const { pathname } = useLocation();
  const section = librarySectionOf(pathname)?.label;
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
