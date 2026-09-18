import { useState } from "react";
import { NavLink, Outlet } from "react-router";

import contractorLogoUrl from "../assets/contractor-logo.png";
import { UI_VERSION } from "../build";
import { useSession } from "../auth/session";
import { Icon } from "./icon";

const navigation = [
  { to: "/", label: "Home", icon: "home", end: true },
  { to: "/projects", label: "Projects", icon: "projects" },
  { to: "/runs", label: "Runs", icon: "runs" },
  { to: "/catalog", label: "Catalog", icon: "catalog" },
  { to: "/artifacts", label: "Artifacts", icon: "artifacts" },
  { to: "/evals", label: "Evals", icon: "evals" },
  { to: "/operations", label: "Operations", icon: "operations" },
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
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
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
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="session-panel">
          <span className="status-dot" aria-hidden="true" />
          <div>
            <strong>{session?.principal.username}</strong>
            <small>UI {UI_VERSION}</small>
            <NavLink
              className="sidebar-utility-link"
              to="/operations/settings"
              onClick={() => setMenuOpen(false)}
            >
              <Icon name="settings" />
              <span>Settings</span>
            </NavLink>
          </div>
          <button
            className="text-button sidebar-utility-link"
            type="button"
            disabled={isLoggingOut}
            onClick={() => void onLogout()}
          >
            <Icon name="logout" />
            <span>{isLoggingOut ? "Signing out…" : "Sign out"}</span>
          </button>
          {logoutError === null ? null : (
            <p className="inline-error" role="alert">
              {logoutError}
            </p>
          )}
        </div>
      </aside>
      <main id="main-content" className="content" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}
