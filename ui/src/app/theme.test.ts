import { readFileSync } from "node:fs";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MemoryStorage } from "../test/storage";

import {
  initializeTheme,
  setThemePreference,
  useResolvedTheme,
  useThemePreference,
} from "./theme";

/** A controllable prefers-color-scheme: dark query. */
function stubSystemTheme(dark: boolean) {
  const listeners = new Set<() => void>();
  const query = {
    matches: dark,
    addEventListener: (_type: string, listener: () => void) =>
      listeners.add(listener),
    removeEventListener: (_type: string, listener: () => void) =>
      listeners.delete(listener),
  };
  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => query),
  );
  return {
    change(next: boolean) {
      query.matches = next;
      for (const listener of listeners) listener();
    },
  };
}

describe("theme preference", () => {
  let meta: HTMLMetaElement;

  beforeEach(() => {
    vi.stubGlobal("localStorage", new MemoryStorage());
    meta = document.createElement("meta");
    meta.name = "color-scheme";
    meta.content = "light dark";
    document.head.append(meta);
  });

  afterEach(() => {
    setThemePreference("system");
    meta.remove();
    vi.unstubAllGlobals();
  });

  it("resolves system to the operating system's setting and follows it", () => {
    const system = stubSystemTheme(true);
    initializeTheme();
    const { result } = renderHook(() => ({
      preference: useThemePreference(),
      resolved: useResolvedTheme(),
    }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(result.current).toEqual({ preference: "system", resolved: "dark" });
    expect(meta.content).toBe("light dark");

    act(() => system.change(false));
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(result.current.resolved).toBe("light");

    // An explicit choice ignores later changes of the system setting.
    act(() => setThemePreference("black"));
    act(() => system.change(true));
    expect(document.documentElement.dataset.theme).toBe("black");
    expect(result.current).toEqual({ preference: "black", resolved: "black" });
  });

  it("applies and stores an explicit choice, and clears it for system", () => {
    stubSystemTheme(false);
    setThemePreference("black");
    expect(document.documentElement.dataset.theme).toBe("black");
    expect(localStorage.getItem("contractor.theme")).toBe("black");
    expect(meta.content).toBe("dark");

    setThemePreference("system");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("contractor.theme")).toBeNull();
    expect(meta.content).toBe("light dark");
  });

  it("reads the stored choice on start", () => {
    stubSystemTheme(false);
    localStorage.setItem("contractor.theme", "dark");
    initializeTheme();
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(meta.content).toBe("dark");
  });

  it("still applies the theme when browser storage is blocked", () => {
    const blocked = new MemoryStorage();
    blocked.setItem = () => {
      throw new DOMException("blocked", "SecurityError");
    };
    vi.stubGlobal("localStorage", blocked);
    setThemePreference("light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(meta.content).toBe("light");
  });
});

/** Light tokens the other palettes inherit: the brand mark and the selection aliases. */
const THEME_INDEPENDENT = new Set([
  "--brand-bg",
  "--sel",
  "--sel-bar",
  "--sel-line",
]);

describe("theme.css", () => {
  const css = readFileSync("src/app/theme.css", "utf8");

  /** Custom properties declared in the first block that opens with `head`. */
  function block(head: string): Map<string, string> {
    const start = css.indexOf(head);
    if (start < 0) throw new Error(`no ${head} block`);
    const body = css.slice(start + head.length, css.indexOf("}", start));
    return new Map(
      Array.from(body.matchAll(/(--[a-z0-9-]+|color-scheme):\s*([^;]+);/g)).map(
        (match) => [match[1] ?? "", (match[2] ?? "").trim()],
      ),
    );
  }

  it("paints the first frame of a dark system theme like the dark palette", () => {
    const firstPaint = block(":root:not([data-theme]) {");
    const dark = block(':root[data-theme="dark"] {');
    expect(firstPaint.size).toBeGreaterThan(0);
    for (const [name, value] of firstPaint) {
      expect(`${name}: ${value}`).toBe(`${name}: ${dark.get(name)}`);
    }
  });

  it("defines every light token in the dark and black palettes", () => {
    const light = block(":root {");
    const themed = Array.from(light.keys()).filter((name) =>
      // The scale and font tokens do not change with the theme.
      /^--(?!space-|radius-|text-|control-|font-)/.test(name),
    );
    for (const palette of [
      ':root[data-theme="dark"] {',
      ':root[data-theme="black"] {',
    ]) {
      const tokens = block(palette);
      const missing = themed.filter(
        (name) => !tokens.has(name) && !THEME_INDEPENDENT.has(name),
      );
      expect(missing, palette).toEqual([]);
    }
  });
});
