import { useSyncExternalStore } from "react";

/** What the user chose. "system" follows the operating system's light or dark setting. */
export type ThemePreference = "system" | "light" | "dark" | "black";

/** The palette actually on screen. */
export type ResolvedTheme = "light" | "dark" | "black";

export const THEME_PREFERENCES: readonly ThemePreference[] = [
  "system",
  "light",
  "dark",
  "black",
];

/**
 * Browser-local appearance preference. It is neither a secret nor a draft, so
 * it may live in localStorage; reads and writes tolerate blocked storage.
 */
export const THEME_STORAGE_KEY = "contractor.theme";

const DARK_QUERY = "(prefers-color-scheme: dark)";

const listeners = new Set<() => void>();
let preference: ThemePreference = "system";

function isThemePreference(value: unknown): value is ThemePreference {
  return (
    typeof value === "string" &&
    (THEME_PREFERENCES as readonly string[]).includes(value)
  );
}

function readStoredPreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

function darkMediaQuery(): MediaQueryList | undefined {
  return typeof window.matchMedia === "function"
    ? window.matchMedia(DARK_QUERY)
    : undefined;
}

export function resolveTheme(
  value: ThemePreference,
  systemPrefersDark = darkMediaQuery()?.matches === true,
): ResolvedTheme {
  if (value === "system") {
    return systemPrefersDark ? "dark" : "light";
  }
  return value;
}

/**
 * Applies a preference to the document. "system" removes the attribute so the
 * stylesheet's prefers-color-scheme rules decide, including live OS changes.
 */
export function applyTheme(
  value: ThemePreference,
  root: HTMLElement = document.documentElement,
): void {
  if (value === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", value);
  }
  const meta = document.querySelector<HTMLMetaElement>(
    'meta[name="color-scheme"]',
  );
  if (meta !== null) {
    meta.content =
      value === "system" ? "light dark" : value === "light" ? "light" : "dark";
  }
}

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

/**
 * Reads the stored preference and applies it. Call once before the first
 * render; the CSP forbids an inline bootstrap script in index.html.
 */
export function initializeTheme(): void {
  preference = readStoredPreference();
  applyTheme(preference);
  window.addEventListener("storage", (event) => {
    if (event.key !== THEME_STORAGE_KEY) {
      return;
    }
    preference = isThemePreference(event.newValue) ? event.newValue : "system";
    applyTheme(preference);
    notify();
  });
  darkMediaQuery()?.addEventListener("change", notify);
}

export function getThemePreference(): ThemePreference {
  return preference;
}

export function setThemePreference(value: ThemePreference): void {
  preference = value;
  try {
    if (value === "system") {
      localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      localStorage.setItem(THEME_STORAGE_KEY, value);
    }
  } catch {
    // Storage can be blocked; the choice still applies to this page.
  }
  applyTheme(value);
  notify();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useThemePreference(): ThemePreference {
  return useSyncExternalStore(subscribe, getThemePreference, () => "system");
}

/** The palette on screen, for components that take a theme prop instead of CSS. */
export function useResolvedTheme(): ResolvedTheme {
  return useSyncExternalStore(
    subscribe,
    () => resolveTheme(preference),
    () => "light",
  );
}
