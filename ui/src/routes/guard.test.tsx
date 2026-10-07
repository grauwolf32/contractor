import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import { describe, expect, it, vi } from "vitest";

import type { AuthSession } from "../api/client";
import { APICompatibilityError, PublicAPIError } from "../api/error";
import { queryKeys } from "../api/query-keys";
import { createApplicationQueryClient } from "../app/query-client";
import { SessionProvider, type SessionAPI } from "../auth/session";
import { AuthenticatedRoute } from "./guard";
import { LoginRoute } from "./login";
import { NotFoundRoute, RouteChunkLoading } from "./placeholders";

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user", "operations"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2026-09-20T12:00:00Z",
  absoluteExpiresAt: "2026-09-21T12:00:00Z",
};

function renderGuard(getSession: SessionAPI["getSession"], initialPath = "/") {
  const queryClient = createApplicationQueryClient();
  const api: SessionAPI = {
    getSession,
    login: vi.fn(async () => session),
    logout: vi.fn(async () => undefined),
  };
  const router = createMemoryRouter(
    [
      {
        element: <AuthenticatedRoute />,
        children: [{ path: "/", element: <h1>Protected page</h1> }],
      },
      { path: "/login", element: <LoginRoute /> },
    ],
    { initialEntries: [initialPath] },
  );
  render(
    <QueryClientProvider client={queryClient}>
      <SessionProvider api={api}>
        <RouterProvider router={router} />
      </SessionProvider>
    </QueryClientProvider>,
  );
  return queryClient;
}

describe("AuthenticatedRoute", () => {
  it("keeps rendering a cached session when a refresh fails and offers a retry", async () => {
    const getSession = vi
      .fn<SessionAPI["getSession"]>()
      .mockResolvedValueOnce(session)
      .mockRejectedValueOnce(new Error("Public API is unavailable"))
      .mockResolvedValue(session);
    const queryClient = renderGuard(getSession);
    await screen.findByRole("heading", { name: "Protected page" });
    await act(() =>
      queryClient.refetchQueries({ queryKey: queryKeys.session }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not refresh the Server session",
    );
    expect(
      screen.getByRole("heading", { name: "Protected page" }),
    ).toBeVisible();
    expect(
      screen.queryByText("Contractor Server is not compatible"),
    ).not.toBeInTheDocument();
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    );
    expect(getSession).toHaveBeenCalledTimes(3);
  });

  it("blocks a cached session when the Server becomes incompatible", async () => {
    const getSession = vi
      .fn<SessionAPI["getSession"]>()
      .mockResolvedValueOnce(session)
      .mockRejectedValue(new APICompatibilityError("contractor.public.v9"));
    const queryClient = renderGuard(getSession);
    await screen.findByRole("heading", { name: "Protected page" });
    await act(() =>
      queryClient.refetchQueries({ queryKey: queryKeys.session }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Contractor Server is not compatible",
      }),
    ).toBeVisible();
    expect(
      screen.queryByRole("heading", { name: "Protected page" }),
    ).not.toBeInTheDocument();
  });

  it("blocks when no session could be loaded", async () => {
    renderGuard(vi.fn(async () => Promise.reject(new Error("offline"))));
    expect(
      await screen.findByRole("heading", {
        name: "Server unavailable",
      }),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Try again" })).toBeEnabled();
  });

  it("recovers an authenticated route after a transient bootstrap failure", async () => {
    const unavailable = new PublicAPIError({
      status: 503,
      code: "server_unavailable",
      message: "Server unavailable",
      retryable: true,
    });
    const getSession = vi
      .fn<SessionAPI["getSession"]>()
      .mockRejectedValueOnce(unavailable)
      .mockResolvedValue(session);
    renderGuard(getSession);
    await screen.findByRole("heading", { name: "Protected page" });
    expect(getSession).toHaveBeenCalledTimes(2);
  });

  it("recovers the login form after a transient bootstrap failure", async () => {
    const unavailable = new PublicAPIError({
      status: 503,
      code: "server_unavailable",
      message: "Server unavailable",
      retryable: true,
    });
    const getSession = vi
      .fn<SessionAPI["getSession"]>()
      .mockRejectedValueOnce(unavailable)
      .mockResolvedValue(null);
    renderGuard(getSession, "/login");
    await waitFor(() => expect(getSession).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled(),
    );
  });

  it.each(["/", "/login"])(
    "offers manual retry on the %s bootstrap error screen",
    async (initialPath) => {
      const getSession = vi
        .fn<SessionAPI["getSession"]>()
        .mockRejectedValueOnce(new Error("offline"))
        .mockResolvedValue(initialPath === "/" ? session : null);
      renderGuard(getSession, initialPath);
      expect(
        await screen.findByRole("heading", { name: "Server unavailable" }),
      ).toBeVisible();
      await userEvent
        .setup()
        .click(screen.getByRole("button", { name: "Try again" }));
      await waitFor(() => expect(getSession).toHaveBeenCalledTimes(2));
      if (initialPath === "/") {
        await screen.findByRole("heading", { name: "Protected page" });
      } else {
        expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled();
      }
    },
  );

  it("never automatically retries an incompatible bootstrap response", async () => {
    const getSession = vi
      .fn<SessionAPI["getSession"]>()
      .mockRejectedValue(new APICompatibilityError("contractor.public.v9"));
    renderGuard(getSession);
    await screen.findByRole("heading", {
      name: "Contractor Server is not compatible",
    });
    await new Promise((resolve) => setTimeout(resolve, 350));
    expect(getSession).toHaveBeenCalledTimes(1);
  });
});

describe("Full-page states", () => {
  it("shows the session check while the Server answers", async () => {
    let answer!: (value: AuthSession) => void;
    renderGuard(
      vi.fn(
        () =>
          new Promise<AuthSession>((resolve) => {
            answer = resolve;
          }),
      ),
    );
    expect(screen.getByText("Checking Server session…")).toBeVisible();
    await act(async () => answer(session));
    expect(
      await screen.findByRole("heading", { name: "Protected page" }),
    ).toBeVisible();
  });

  it("offers Inbox and Projects from a page that does not exist", async () => {
    const router = createMemoryRouter(
      [{ path: "*", element: <NotFoundRoute /> }],
      { initialEntries: ["/no/such/page"] },
    );
    render(<RouterProvider router={router} />);
    expect(
      await screen.findByRole("heading", { level: 1, name: "Page not found" }),
    ).toBeVisible();
    const recovery = screen.getByRole("navigation", { name: "Recovery" });
    expect(
      within(recovery)
        .getAllByRole("link")
        .map((link) => [link.textContent, link.getAttribute("href")]),
    ).toEqual([
      ["Inbox", "/"],
      ["Projects", "/projects"],
    ]);
    expect(document.title).toBe("Page not found · Contractor");
  });

  it("announces the first route chunk while it loads", () => {
    render(<RouteChunkLoading />);
    expect(screen.getByText("Loading workspace…")).toBeVisible();
    expect(screen.getByRole("main")).toHaveAttribute("aria-live", "polite");
  });
});
