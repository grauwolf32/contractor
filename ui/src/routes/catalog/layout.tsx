import { Navigate, NavLink, Outlet, useLocation, useMatch } from "react-router";

import "./catalog.css";

export function CatalogIndexRedirect() {
  const { search, hash } = useLocation();
  return (
    <Navigate replace to={{ pathname: "/catalog/workflows", search, hash }} />
  );
}

export function CatalogLayoutRoute() {
  const workflowDetail = useMatch("/catalog/workflows/:name/:version");
  if (workflowDetail)
    return (
      <div className="catalog-page">
        <Outlet />
      </div>
    );
  return (
    <div className="catalog-page">
      <header>
        <h2>Catalog</h2>
      </header>
      <nav
        className="catalog-navigation section-navigation"
        aria-label="Catalog navigation"
      >
        <NavLink to="/catalog/workflows">Workflows</NavLink>
        <NavLink to="/catalog/agents">Agents</NavLink>
        <NavLink to="/catalog/skills">Skills</NavLink>
      </nav>
      <Outlet />
    </div>
  );
}
