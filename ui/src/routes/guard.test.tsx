import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import { describe, expect, it, vi } from "vitest";

import type { AuthSession } from "../api/client";
import { APICompatibilityError } from "../api/error";
import { queryKeys } from "../api/query-keys";
import { createApplicationQueryClient } from "../app/query-client";
import { SessionProvider, type SessionAPI } from "../auth/session";
import { AuthenticatedRoute } from "./guard";

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

function renderGuard(getSession: SessionAPI["getSession"]) {
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
      { path: "/login", element: <h1>Sign in</h1> },
    ],
    { initialEntries: ["/"] },
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
      screen.queryByText("Contractor Server is not compatible or unavailable"),
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
        name: "Contractor Server is not compatible or unavailable",
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
        name: "Contractor Server is not compatible or unavailable",
      }),
    ).toBeVisible();
  });
});
