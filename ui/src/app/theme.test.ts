import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MemoryStorage } from "../test/storage";

import {
  applyTheme,
  getThemePreference,
  resolveTheme,
  setThemePreference,
  THEME_STORAGE_KEY,
} from "./theme";

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

  it("resolves system to the operating system's light or dark setting", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("black", false)).toBe("black");
    expect(resolveTheme("light", true)).toBe("light");
  });

  it("applies and stores an explicit choice, and clears it for system", () => {
    setThemePreference("black");
    expect(document.documentElement.dataset.theme).toBe("black");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("black");
    expect(meta.content).toBe("dark");
    expect(getThemePreference()).toBe("black");

    setThemePreference("system");
    expect(document.documentElement.hasAttribute("data-theme")).toBe(false);
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
    expect(meta.content).toBe("light dark");
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

  it("sets the color-scheme meta for native controls", () => {
    applyTheme("dark");
    expect(meta.content).toBe("dark");
    applyTheme("light");
    expect(meta.content).toBe("light");
  });
});
