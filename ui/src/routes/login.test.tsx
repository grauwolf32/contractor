import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";
import { RouterProvider } from "react-router/dom";
import { describe, expect, it, vi } from "vitest";

import type { AuthSession } from "../api/client";
import { createApplicationQueryClient } from "../app/query-client";
import { SessionProvider, type SessionAPI } from "../auth/session";
import { LoginRoute } from "./login";

const session: AuthSession = {
  principal: {
    userId: "user_local",
    username: "owner",
    capabilities: ["user"],
  },
  csrfToken: "a".repeat(43),
  idleExpiresAt: "2099-09-20T12:00:00Z",
  absoluteExpiresAt: "2099-09-21T12:00:00Z",
};

function renderLogin(login: SessionAPI["login"], from?: string) {
  const api: SessionAPI = {
    getSession: vi.fn(async () => null),
    login,
    logout: vi.fn(async () => undefined),
  };
  const router = createMemoryRouter(
    [
      { path: "/login", element: <LoginRoute /> },
      { path: "/", element: <h1>Inbox page</h1> },
      { path: "/operations", element: <h1>Operations page</h1> },
    ],
    {
      initialEntries: [
        from === undefined ? "/login" : { pathname: "/login", state: { from } },
      ],
    },
  );
  render(
    <QueryClientProvider client={createApplicationQueryClient()}>
      <SessionProvider api={api}>
        <RouterProvider router={router} />
      </SessionProvider>
    </QueryClientProvider>,
  );
  return router;
}

describe("LoginRoute", () => {
  it("names the product and the UI version and clears the password after a refusal", async () => {
    const login = vi.fn<SessionAPI["login"]>(async () => {
      throw new Error("Username or password is incorrect.");
    });
    renderLogin(login);
    const region = await screen.findByRole("region", { name: "Sign in" });
    expect(region).toHaveTextContent("Contractor");
    expect(region).toHaveTextContent(/UI \d+\.\d+\.\d+/);
    const user = userEvent.setup();
    await user.type(screen.getByLabelText("Username"), "owner");
    await user.type(screen.getByLabelText("Password"), "wrong-secret");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Username or password is incorrect.",
    );
    expect(screen.getByLabelText("Password")).toHaveValue("");
    expect(screen.getByLabelText("Username")).toHaveValue("owner");
    expect(screen.getByLabelText("Password")).toHaveAttribute(
      "autocomplete",
      "current-password",
    );
    expect(login).toHaveBeenCalledWith({
      username: "owner",
      password: "wrong-secret",
    });
    expect(document.body).not.toHaveTextContent("wrong-secret");
  });

  it.each([
    ["/operations", "Operations page"],
    ["//evil.example/steal", "Inbox page"],
    ["https://evil.example/", "Inbox page"],
  ])("returns from %s only to a same-site path", async (from, heading) => {
    const router = renderLogin(
      vi.fn(async () => session),
      from,
    );
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("Username"), "owner");
    await user.type(screen.getByLabelText("Password"), "secret");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(
      await screen.findByRole("heading", { name: heading }),
    ).toBeInTheDocument();
    expect(router.state.location.pathname).toBe(
      heading === "Inbox page" ? "/" : from,
    );
  });
});
