import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { UI_VERSION } from "../build";
import { sessionRunDraftStore } from "../run-drafts/session-stores";
import { MemoryStorage } from "../test/storage";
import { renderShell, TEST_USERNAME } from "./shell-test-harness";
import { setThemePreference } from "./theme";

async function openMenu() {
  const user = userEvent.setup();
  const button = await screen.findByRole("button", { name: "Account" });
  await user.click(button);
  return { user, button };
}

describe("account menu", () => {
  beforeEach(() => {
    vi.stubGlobal("localStorage", new MemoryStorage());
  });

  afterEach(() => {
    setThemePreference("system");
    vi.unstubAllGlobals();
  });

  it("shows the user's initial and opens on the first item", async () => {
    renderShell("/runs");
    const button = await screen.findByRole("button", { name: "Account" });
    expect(button).toHaveTextContent(/^O$/);
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByRole("button", { name: "Sign out" }),
    ).not.toBeInTheDocument();

    const { user } = await openMenu();
    expect(button).toHaveAttribute("aria-expanded", "true");
    const menu = document.getElementById(
      button.getAttribute("aria-controls") ?? "",
    );
    expect(menu).not.toBeNull();
    const panel = within(menu as HTMLElement);
    expect(panel.getByText(TEST_USERNAME)).toBeVisible();
    const settings = panel.getByRole("link", { name: "Settings" });
    expect(settings).toHaveAttribute("href", "/operations/settings");
    expect(settings).toHaveFocus();
    expect(panel.getByRole("button", { name: "Sign out" })).toBeEnabled();
    expect(panel.getByText(`UI ${UI_VERSION}`)).toBeVisible();
    expect(
      panel
        .getAllByRole("radio")
        .map((radio) => radio.parentElement?.textContent),
    ).toEqual(["System", "Light", "Dark", "Black"]);

    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "false");
  });

  it("closes on Escape and returns focus to the button", async () => {
    renderShell("/runs");
    const { user, button } = await openMenu();
    await user.keyboard("{Escape}");
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(button).toHaveFocus();
    expect(
      screen.queryByRole("link", { name: "Settings" }),
    ).not.toBeInTheDocument();
  });

  it("closes on an outside click and returns focus to the button", async () => {
    renderShell("/runs");
    const { user, button } = await openMenu();
    await user.click(screen.getByText("Page at /runs"));
    expect(button).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(button).toHaveFocus());
  });

  it("leaves focus on a control the outside click chose", async () => {
    renderShell("/runs");
    const { user, button } = await openMenu();
    const runs = screen.getByRole("link", { name: "Runs" });
    await user.pointer({ keys: "[MouseLeft>]", target: runs });
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(runs).toHaveFocus();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(runs).toHaveFocus();
  });

  it("closes when focus leaves it", async () => {
    renderShell("/runs");
    const { user, button } = await openMenu();
    expect(screen.getByRole("link", { name: "Settings" })).toHaveFocus();
    await user.tab({ shift: true });
    expect(button).toHaveFocus();
    expect(button).toHaveAttribute("aria-expanded", "true");
    await user.tab({ shift: true });
    expect(button).toHaveAttribute("aria-expanded", "false");
  });

  it("closes after following Settings", async () => {
    const { router } = renderShell("/runs");
    const { user, button } = await openMenu();
    await user.click(screen.getByRole("link", { name: "Settings" }));
    expect(
      await screen.findByText("Page at /operations/settings"),
    ).toBeVisible();
    expect(router.state.location.pathname).toBe("/operations/settings");
    expect(button).toHaveAttribute("aria-expanded", "false");
  });

  it("sets the theme from the radio group", async () => {
    renderShell("/runs");
    const { user } = await openMenu();
    const theme = screen.getByRole("group", { name: "Theme" });
    expect(within(theme).getByRole("radio", { name: "System" })).toBeChecked();

    await user.click(within(theme).getByRole("radio", { name: "Dark" }));
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    expect(within(theme).getByRole("radio", { name: "Dark" })).toBeChecked();

    await user.click(within(theme).getByRole("radio", { name: "Black" }));
    expect(document.documentElement).toHaveAttribute("data-theme", "black");

    await user.click(within(theme).getByRole("radio", { name: "System" }));
    expect(document.documentElement).not.toHaveAttribute("data-theme");
  });

  it("signs out and discards the session's run drafts", async () => {
    let finish: () => void = () => undefined;
    const logout = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const drafts = sessionRunDraftStore("user_local");
    const { sessionAPI } = renderShell("/runs", { logout });
    const { user } = await openMenu();
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    const pending = await screen.findByRole("button", {
      name: "Signing out…",
    });
    expect(pending).toBeDisabled();
    expect(sessionAPI.logout).toHaveBeenCalledTimes(1);
    expect(sessionRunDraftStore("user_local")).toBe(drafts);

    finish();
    await waitFor(() =>
      expect(sessionRunDraftStore("user_local")).not.toBe(drafts),
    );
  });

  it("shows why signing out failed", async () => {
    const logout = vi.fn(async () => {
      throw new Error("Server unavailable");
    });
    const drafts = sessionRunDraftStore("user_local");
    renderShell("/runs", { logout });
    const { user } = await openMenu();
    await user.click(screen.getByRole("button", { name: "Sign out" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Server unavailable",
    );
    expect(screen.getByRole("button", { name: "Sign out" })).toBeEnabled();
    expect(sessionRunDraftStore("user_local")).toBe(drafts);
  });
});
