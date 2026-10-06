import "./shell.css";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, Outlet, useLocation } from "react-router";

import contractorLogoUrl from "../assets/contractor-logo.png";
import { useInboxSummary } from "../api/cross-project";
import { useSession } from "../auth/session";
import { isApplePlatform, Kbd, modKeyLabel, useShortcuts } from "../ui";
import { AccountMenu, AccountPanel } from "./account-menu";
import { CommandPalette } from "./command-palette";
import {
  activeDestination,
  type Destination,
  type DestinationId,
  destinationsFor,
} from "./destinations";
import { Icon } from "./icon";

const NAVIGATION_ID = "primary-navigation";
/** Above this width the rail shows and the phone drawer has no use. */
const RAIL_QUERY = "(min-width: 821px)";

interface Badge {
  /** Visible count, capped at "99+". */
  text: string;
  /** What the count means, for screen readers. */
  description: string;
}

function RailLink({
  destination,
  active,
  badge,
  onNavigate,
}: {
  destination: Destination;
  active: boolean;
  badge?: Badge | undefined;
  onNavigate: () => void;
}) {
  return (
    <li>
      <Link
        className="shell-nav-link"
        to={destination.to}
        aria-current={active ? "page" : undefined}
        onClick={onNavigate}
      >
        <span className="shell-nav-icon" aria-hidden="true">
          <Icon name={destination.icon} />
          {badge === undefined ? null : (
            <span className="shell-nav-badge">{badge.text}</span>
          )}
        </span>
        <span className="shell-nav-label">{destination.label}</span>
        {badge === undefined ? null : (
          <>
            {/* Whitespace between flex items draws nothing but keeps the
                words apart in the link's name: "Inbox (3 need your decision)". */}{" "}
            <span className="ui-visually-hidden">({badge.description})</span>
          </>
        )}
      </Link>
    </li>
  );
}

/** The Inbox link with the number of things waiting for the user. */
function InboxRailLink(props: {
  destination: Destination;
  active: boolean;
  onNavigate: () => void;
}) {
  const { needsDecision } = useInboxSummary();
  const badge =
    needsDecision === undefined || needsDecision <= 0
      ? undefined
      : {
          text: needsDecision > 99 ? "99+" : String(needsDecision),
          description: `${needsDecision} ${needsDecision === 1 ? "needs" : "need"} your decision`,
        };
  return <RailLink {...props} badge={badge} />;
}

function RailGroup({
  destinations,
  active,
  onNavigate,
  className = "shell-nav-group",
}: {
  destinations: readonly Destination[];
  active: DestinationId | undefined;
  onNavigate: () => void;
  className?: string;
}) {
  return (
    <ul className={className}>
      {destinations.map((destination) => {
        const props = {
          destination,
          active: destination.id === active,
          onNavigate,
        };
        return destination.id === "inbox" ? (
          <InboxRailLink key={destination.id} {...props} />
        ) : (
          <RailLink key={destination.id} {...props} />
        );
      })}
    </ul>
  );
}

/** The top bar's "Search or start a check…" field that opens the palette. */
function CommandButton({ onOpen }: { onOpen: () => void }) {
  return (
    <button
      className="shell-command"
      type="button"
      aria-keyshortcuts={isApplePlatform() ? "Meta+K" : "Control+K"}
      onClick={onOpen}
    >
      <Icon name="search" />
      <span className="shell-command-text">Search or start a check…</span>
      <span className="ui-kbd-hint" aria-hidden="true">
        <Kbd>{modKeyLabel()}</Kbd>
        <Kbd>K</Kbd>
      </span>
    </button>
  );
}

/**
 * The V3B application frame (docs/design/ui/v3b-build-contract.md §4, §6):
 * the rail with the primary navigation and the account menu, a top bar with
 * the command button, and the page. At 820 px and below a top bar with the
 * logo, a command button and a Menu toggle replaces the rail, and the same
 * navigation opens as a drawer that also holds the account items.
 */
export function ApplicationShell({ error }: { error?: ReactNode }) {
  const { session } = useSession();
  const location = useLocation();
  const active = activeDestination(location.pathname);
  const destinations = destinationsFor(session?.principal.capabilities);
  // Open for the location it was opened at, so any navigation closes it.
  const [menuOpenAt, setMenuOpenAt] = useState<string | null>(null);
  // Forget that location once it is left: Back and Forward restore entry
  // keys, and returning to the entry must not reopen the drawer.
  if (menuOpenAt !== null && menuOpenAt !== location.key) setMenuOpenAt(null);
  const menuOpen = menuOpenAt === location.key;
  const [paletteOpen, setPaletteOpen] = useState(false);
  const toggle = useRef<HTMLButtonElement>(null);
  const navigation = useRef<HTMLElement>(null);
  const closeMenu = () => setMenuOpenAt(null);

  useShortcuts(
    {
      escape: () => {
        const focusInMenu =
          navigation.current?.contains(document.activeElement) ?? false;
        setMenuOpenAt(null);
        if (focusInMenu) toggle.current?.focus();
      },
    },
    { enabled: menuOpen },
  );

  // Widening the window past the phone layout closes the drawer.
  useEffect(() => {
    if (!menuOpen || typeof window.matchMedia !== "function") return undefined;
    const rail = window.matchMedia(RAIL_QUERY);
    const onChange = () => {
      if (rail.matches) setMenuOpenAt(null);
    };
    rail.addEventListener("change", onChange);
    return () => rail.removeEventListener("change", onChange);
  }, [menuOpen]);

  const group = (name: Destination["group"]) =>
    destinations.filter((destination) => destination.group === name);
  const bottom = group("bottom");

  return (
    <div className="application" data-menu-open={menuOpen}>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <div className="shell-rail">
        <div className="shell-bar">
          <Link className="shell-brand" to="/" onClick={closeMenu}>
            <img src={contractorLogoUrl} alt="Contractor" />
          </Link>
          <button
            className="shell-bar-button shell-bar-search"
            type="button"
            aria-label="Search or start a check"
            onClick={() => setPaletteOpen(true)}
          >
            <Icon name="search" />
          </button>
          <button
            ref={toggle}
            className="shell-bar-button shell-menu-toggle"
            type="button"
            aria-expanded={menuOpen}
            aria-controls={NAVIGATION_ID}
            onClick={() => setMenuOpenAt(menuOpen ? null : location.key)}
          >
            <Icon name={menuOpen ? "close" : "menu"} />
            <span>{menuOpen ? "Close menu" : "Menu"}</span>
          </button>
        </div>
        <nav
          ref={navigation}
          id={NAVIGATION_ID}
          className="shell-nav"
          aria-label="Primary navigation"
        >
          <RailGroup
            destinations={group("main")}
            active={active}
            onNavigate={closeMenu}
          />
          <RailGroup
            className="shell-nav-group shell-nav-more"
            destinations={group("more")}
            active={active}
            onNavigate={closeMenu}
          />
          <div className="shell-nav-bottom">
            {bottom.length === 0 ? null : (
              <RailGroup
                destinations={bottom}
                active={active}
                onNavigate={closeMenu}
              />
            )}
            <AccountMenu />
          </div>
          {menuOpen ? (
            <div className="shell-drawer-account">
              <AccountPanel onNavigate={closeMenu} />
            </div>
          ) : null}
        </nav>
      </div>
      {/* The open drawer covers the page; Tab must not reach what it hides. */}
      <div className="shell-main" inert={menuOpen}>
        <div className="shell-topbar">
          <CommandButton onOpen={() => setPaletteOpen(true)} />
        </div>
        <main id="main-content" className="content" tabIndex={-1}>
          {error ?? <Outlet />}
        </main>
      </div>
      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}
