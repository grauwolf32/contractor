import { Navigate, NavLink, Outlet, useLocation } from "react-router";

import "./catalog.css";

export function LegacyCatalogRedirect() {
  const { pathname, search, hash } = useLocation();
  return (
    <Navigate replace to={{ pathname: `/catalog${pathname}`, search, hash }} />
  );
}

export function CatalogIndexRedirect() {
  const { search, hash } = useLocation();
  return (
    <Navigate replace to={{ pathname: "/catalog/workflows", search, hash }} />
  );
}

export function CatalogLayoutRoute() {
  return (
    <div className="catalog-page">
      <header>
        <p className="eyebrow">Reusable execution definitions</p>
        <h2>Catalog</h2>
      </header>
      <nav className="catalog-navigation" aria-label="Catalog navigation">
        <NavLink to="/catalog/workflows">Workflows</NavLink>
        <NavLink to="/catalog/agents">Agents</NavLink>
        <NavLink to="/catalog/skills">Skills</NavLink>
      </nav>
      <Outlet />
    </div>
  );
}
