import { useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router";

import contractorLogoUrl from "../assets/contractor-logo.png";
import { UI_VERSION } from "../build";
import { useSession } from "../auth/session";
import { discardSessionRunDrafts } from "../run-drafts/session-stores";
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
  const { pathname } = useLocation();
  const skillDetail = pathname.startsWith("/artifacts/skills/");
  const navigationActive = (to: string) =>
    skillDetail
      ? to === "/catalog"
      : to === "/"
        ? pathname === "/"
        : pathname === to || pathname.startsWith(`${to}/`);
  const [logoutError, setLogoutError] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  async function onLogout() {
    setLogoutError(null);
    try {
      await logout();
      discardSessionRunDrafts();
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
            <Link
              key={item.to}
              to={item.to}
              className={navigationActive(item.to) ? "active" : undefined}
              aria-current={navigationActive(item.to) ? "page" : undefined}
              onClick={() => setMenuOpen(false)}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
        <div className="session-panel">
          <div className="session-identity">
            <span className="status-dot" aria-hidden="true" />
            <strong title={session?.principal.username}>
              {session?.principal.username}
            </strong>
          </div>
          <div className="session-actions">
            <NavLink
              className={({ isActive }) =>
                `sidebar-utility-link${isActive ? " active" : ""}`
              }
              to="/operations/settings"
              onClick={() => setMenuOpen(false)}
            >
              <Icon name="settings" />
              <span>Settings</span>
            </NavLink>
            <button
              className="sidebar-utility-link"
              type="button"
              disabled={isLoggingOut}
              onClick={() => void onLogout()}
            >
              <Icon name="logout" />
              <span>{isLoggingOut ? "Signing out…" : "Sign out"}</span>
            </button>
          </div>
          {logoutError === null ? null : (
            <p className="inline-error" role="alert">
              {logoutError}
            </p>
          )}
          <small className="session-version">UI {UI_VERSION}</small>
        </div>
      </aside>
      <main id="main-content" className="content" tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}
