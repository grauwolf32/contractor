import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { AuthSession } from "../api/client";
import type { SessionAPI } from "../auth/session";
import { Application } from "./application";
import { applicationRoutes } from "./router";

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-08-31T20:00:00Z",
  absoluteExpiresAt: "2026-09-01T12:00:00Z",
};

function renderApplication(api: SessionAPI, initialPath: string) {
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [initialPath],
  });
  const view = render(<Application api={api} router={router} />);
  return { ...view, router };
}

describe("application session shell", () => {
  it("guards domain routes with the local login", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => null),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    renderApplication(api, "/runs");
    expect(
      await screen.findByRole("heading", {
        name: "Open the control workspace",
      }),
    ).toBeInTheDocument();
  });

  it("renders guarded navigation from an authoritative session", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => session),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    renderApplication(api, "/runs");
    expect(
      await screen.findByRole("heading", { name: "Runs" }),
    ).toBeInTheDocument();
    expect(screen.getByText("owner")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Operations" })).toHaveAttribute(
      "href",
      "/operations",
    );
  });

  it("shows session bootstrap incompatibility on the login route", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => {
        throw new Error("Server public API version is not supported");
      }),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    renderApplication(api, "/login");
    expect(
      await screen.findByRole("heading", {
        name: "Contractor Server is not compatible or unavailable",
      }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it("logs in and clears the transient password field", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => null),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    const { router } = renderApplication(api, "/login");
    const user = userEvent.setup();
    await screen.findByRole("heading", { name: "Open the control workspace" });
    await user.type(screen.getByLabelText("Username"), "owner");
    await user.type(screen.getByLabelText("Password"), "a-long-local-password");
    await user.click(screen.getByRole("button", { name: "Sign in" }));

    await waitFor(() =>
      expect(router.state.location.pathname).toBe("/workflows"),
    );
    expect(api.login).toHaveBeenCalledWith({
      username: "owner",
      password: "a-long-local-password",
    });
  });
});
