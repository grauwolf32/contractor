import { useState } from "react";
import { NavLink, Outlet } from "react-router";

import contractorLogoUrl from "../assets/contractor-logo.png";
import { UI_VERSION } from "../build";
import { useSession } from "../auth/session";

const navigation = [
  { to: "/", label: "Home", end: true },
  { to: "/projects", label: "Projects" },
  { to: "/runs", label: "Runs" },
  { to: "/catalog", label: "Catalog" },
  { to: "/artifacts", label: "Artifacts" },
  { to: "/evals", label: "Evals" },
  { to: "/operations", label: "Operations" },
] as const;

export function ApplicationShell() {
  const { session, logout, isLoggingOut } = useSession();
  const [logoutError, setLogoutError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  async function onLogout() {
    setLogoutError(null);
    try {
      await logout();
    } catch (error) {
      setLogoutError(
        error instanceof Error ? error.message : "Could not end the session",
      );
    }
  }

  return (
    <div className="application">
      <aside className="sidebar" data-menu-open={menuOpen}>
        <div>
          <img
            className="brand-mark"
            src={contractorLogoUrl}
            alt=""
            aria-hidden="true"
          />
          <p className="eyebrow">Contractor</p>
          <h1>Control workspace</h1>
        </div>
        <button
          className="secondary-button mobile-nav-toggle"
          type="button"
          aria-expanded={menuOpen}
          aria-controls="primary-navigation"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? "Close menu" : "Menu"}
        </button>
        <nav id="primary-navigation" aria-label="Primary navigation">
          {navigation.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={"end" in item ? item.end : false}
              className={({ isActive }) => (isActive ? "active" : undefined)}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="session-panel">
          <span className="status-dot" aria-hidden="true" />
          <div>
            <strong>{session?.principal.username}</strong>
            <small>UI {UI_VERSION}</small>
            <NavLink to="/settings">Personal settings</NavLink>
          </div>
          <button
            className="text-button"
            type="button"
            disabled={isLoggingOut}
            onClick={() => void onLogout()}
          >
            {isLoggingOut ? "Signing out…" : "Sign out"}
          </button>
          {logoutError === null ? null : (
            <p className="inline-error" role="alert">
              {logoutError}
            </p>
          )}
        </div>
      </aside>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
