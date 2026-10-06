import { useEffect, useId, useRef, useState } from "react";
import { Link, useLocation } from "react-router";

import { useSession } from "../auth/session";
import { UI_VERSION } from "../build";
import { discardSessionRunDrafts } from "../run-drafts/session-stores";
import { useShortcuts } from "../ui";
import { Icon } from "./icon";
import {
  setThemePreference,
  THEME_PREFERENCES,
  type ThemePreference,
  useThemePreference,
} from "./theme";

const SETTINGS_PATH = "/operations/settings";

const THEME_LABELS: Readonly<Record<ThemePreference, string>> = {
  system: "System",
  light: "Light",
  dark: "Dark",
  black: "Black",
};

/** Upper-case first letter of the user name, for the avatar. */
function initialOf(username: string | undefined): string | undefined {
  const first = Array.from(username?.trim() ?? "")[0];
  return first === undefined ? undefined : first.toLocaleUpperCase();
}

function Avatar({ username }: { username: string | undefined }) {
  const initial = initialOf(username);
  return initial === undefined ? <Icon name="user" /> : <>{initial}</>;
}

export interface AccountPanelProps {
  /** Runs when a link in the panel is followed, e.g. to close the menu. */
  onNavigate?: (() => void) | undefined;
}

/**
 * What the account menu holds: the signed-in user, Settings, the theme
 * choice, Sign out and the UI version. The rail's popover and the phone
 * drawer both render it.
 */
export function AccountPanel({ onNavigate }: AccountPanelProps) {
  const { session, logout, isLoggingOut } = useSession();
  const { pathname } = useLocation();
  const username = session?.principal.username;
  const preference = useThemePreference();
  const themeGroup = useId();
  const [logoutError, setLogoutError] = useState<string | null>(null);

  async function signOut() {
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
    <div className="shell-account-panel">
      <div className="shell-account-identity">
        <span className="shell-avatar" aria-hidden="true">
          <Avatar username={username} />
        </span>
        <span className="shell-account-who">
          <span className="shell-account-caption">Signed in as</span>
          <strong title={username}>{username}</strong>
        </span>
      </div>
      <Link
        className="shell-account-item"
        to={SETTINGS_PATH}
        aria-current={pathname === SETTINGS_PATH ? "page" : undefined}
        onClick={onNavigate}
      >
        <Icon name="settings" />
        <span>Settings</span>
      </Link>
      <fieldset className="shell-theme">
        <legend>Theme</legend>
        <div className="shell-theme-options">
          {THEME_PREFERENCES.map((option) => (
            <label key={option} className="shell-theme-option">
              <input
                type="radio"
                name={themeGroup}
                value={option}
                checked={preference === option}
                onChange={() => setThemePreference(option)}
              />
              <span>{THEME_LABELS[option]}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <button
        className="shell-account-item"
        type="button"
        disabled={isLoggingOut}
        onClick={() => void signOut()}
      >
        <Icon name="logout" />
        <span>{isLoggingOut ? "Signing out…" : "Sign out"}</span>
      </button>
      {logoutError === null ? null : (
        <p className="inline-error shell-account-error" role="alert">
          {logoutError}
        </p>
      )}
      <p className="shell-account-version">UI {UI_VERSION}</p>
    </div>
  );
}

const FOCUSABLE =
  'a[href], button:not(:disabled), input:not(:disabled), [tabindex]:not([tabindex="-1"])';

/** Controls a click can focus on purpose (not containers like <main tabindex="-1">). */
const CONTROL = `${FOCUSABLE}, select, textarea, summary, [contenteditable]:not([contenteditable="false"])`;

/**
 * The account button at the bottom of the rail and its popover. Opening
 * focuses the first item; Escape, a click outside and following a link close
 * it and return focus to the button (a click outside leaves focus on the
 * control it chose); any other navigation closes it.
 */
export function AccountMenu() {
  const { session } = useSession();
  const location = useLocation();
  const username = session?.principal.username;
  // Open for the location it was opened at, so any navigation closes it.
  const [openAt, setOpenAt] = useState<string | null>(null);
  // Forget that location once it is left: Back and Forward restore entry
  // keys, and returning to the entry must not reopen the menu.
  if (openAt !== null && openAt !== location.key) setOpenAt(null);
  const open = openAt === location.key;
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const popover = useRef<HTMLDivElement>(null);
  const popoverId = useId();

  useEffect(() => {
    if (!open) return;
    popover.current?.querySelector<HTMLElement>(FOCUSABLE)?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    function onPointerDown(event: PointerEvent) {
      if (event.target instanceof Node && root.current?.contains(event.target))
        return;
      setOpenAt(null);
      // After the click has moved focus: back to the button unless the click
      // focused a control. Closing unsubscribes this listener but must not
      // cancel the timer; the ref is null once the shell is gone.
      window.setTimeout(() => {
        const active = document.activeElement;
        const chosen =
          active instanceof HTMLElement &&
          active.isConnected &&
          active.matches(CONTROL);
        if (!chosen) button.current?.focus();
      });
    }
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  useShortcuts(
    {
      escape: () => {
        setOpenAt(null);
        button.current?.focus();
      },
    },
    { enabled: open },
  );

  return (
    <div
      className="shell-account"
      ref={root}
      onBlur={(event) => {
        // Tabbing out closes the popover so it never covers the focused
        // control; a dialog opened on top (the command palette) keeps it.
        const next = event.relatedTarget;
        if (
          open &&
          next instanceof Element &&
          !event.currentTarget.contains(next) &&
          next.closest("[data-contractor-dialog-layer]") === null
        ) {
          setOpenAt(null);
        }
      }}
    >
      <button
        ref={button}
        className="shell-account-button"
        type="button"
        aria-label="Account"
        aria-expanded={open}
        aria-controls={open ? popoverId : undefined}
        title={username}
        onClick={() => setOpenAt(open ? null : location.key)}
      >
        <Avatar username={username} />
      </button>
      {open ? (
        <div ref={popover} id={popoverId} className="shell-account-popover">
          <AccountPanel
            onNavigate={() => {
              setOpenAt(null);
              // The followed link leaves with the popover; as on Escape,
              // focus goes back to the button instead of the page start.
              button.current?.focus();
            }}
          />
        </div>
      ) : null}
    </div>
  );
}
