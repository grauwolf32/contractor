import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, type RouteObject } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { PublicAPI, type AuthSession } from "../api/client";
import { APICompatibilityError } from "../api/error";
import { UI_VERSION } from "../build";
import type { RuntimeConfig } from "../config/runtime-config";
import type { SessionAPI } from "../auth/session";
import { Application } from "./application";
import { applicationRoutes, lazyRoute } from "./router";

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

const runtimeConfig: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

function response(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    headers: {
      "content-type": "application/json",
      "X-Contractor-API-Version": "contractor.public.v1",
    },
  });
}

function renderApplication(
  api: SessionAPI,
  initialPath: string,
  routes = applicationRoutes(),
) {
  const publicAPI = new PublicAPI(
    runtimeConfig,
    vi.fn(async () => response({ items: [], page: { hasMore: false } })),
  );
  const router = createMemoryRouter(routes, {
    initialEntries: [initialPath],
  });
  const view = render(
    <Application api={api} publicAPI={publicAPI} router={router} />,
  );
  return { ...view, router };
}

function runRoute(routes: RouteObject[]): RouteObject {
  const route = routes[1]?.children?.[0]?.children?.find(
    (candidate) => candidate.path === "/runs",
  );
  if (route === undefined) throw new Error("Runs route is missing");
  return route;
}

function ThrowingRoute(): never {
  throw new Error("Route render failed");
}

describe("application session shell", () => {
  it("shows an in-shell reload action when a lazy route chunk fails", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => session),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    const routes = applicationRoutes();
    runRoute(routes).lazy = lazyRoute(
      async (): Promise<{ RunsRoute: () => null }> => {
        throw new TypeError("Failed to fetch dynamically imported module");
      },
      "RunsRoute",
    );
    renderApplication(api, "/projects", routes);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("link", { name: "Runs" }));
    const message = await screen.findByRole("alert");
    expect(message).toHaveTextContent("This page needs a reload");
    expect(
      screen.getByRole("link", { name: "Reload application" }),
    ).toHaveAttribute("href", "/runs");
    expect(
      screen.getByRole("navigation", { name: "Primary navigation" }),
    ).toBeVisible();
    await user.click(screen.getByRole("link", { name: "Projects" }));
    expect(
      await screen.findByRole("heading", { name: "Projects" }),
    ).toBeVisible();
    await user.click(screen.getByRole("link", { name: "Runs" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This page needs a reload",
    );
  });

  it("keeps the shell visible when a child route throws while rendering", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => session),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    const routes = applicationRoutes();
    const route = runRoute(routes);
    delete route.lazy;
    route.element = <ThrowingRoute />;
    renderApplication(api, "/runs", routes);
    expect(await screen.findByText("This page could not open")).toBeVisible();
    expect(
      screen.getByRole("navigation", { name: "Primary navigation" }),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "Go home" })).toHaveAttribute(
      "href",
      "/",
    );
  });

  it("guards domain routes with the local login", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => null),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    renderApplication(api, "/runs");
    expect(
      await screen.findByRole("region", { name: "Sign in" }),
    ).toBeInTheDocument();
    expect(screen.getByText(`UI ${UI_VERSION}`)).toBeVisible();
    const username = screen.getByLabelText("Username") as HTMLInputElement;
    expect(() => new RegExp(username.pattern, "v")).not.toThrow();
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
    expect(screen.getByRole("link", { name: "Projects" })).toHaveAttribute(
      "href",
      "/projects",
    );
    expect(screen.getByRole("link", { name: "Operations" })).toHaveAttribute(
      "href",
      "/operations",
    );
  });

  it("shows session bootstrap incompatibility on the login route", async () => {
    const api: SessionAPI = {
      getSession: vi.fn(async () => {
        throw new APICompatibilityError("contractor.public.v9");
      }),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    renderApplication(api, "/login");
    expect(
      await screen.findByRole("heading", {
        name: "Contractor Server is not compatible",
      }),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
  });

  it.each(["a-long-local-password", "пароль", "🔐".repeat(256)])(
    "logs in with a valid UTF-8 password without character-count rejection (%#)",
    async (password) => {
      const api: SessionAPI = {
        getSession: vi.fn(async () => null),
        login: vi.fn(async () => session),
        logout: vi.fn(async () => undefined),
      };
      const { router } = renderApplication(api, "/login");
      const user = userEvent.setup();
      await screen.findByRole("region", { name: "Sign in" });
      await user.type(screen.getByLabelText("Username"), "owner");
      const passwordInput = screen.getByLabelText("Password");
      expect(passwordInput).not.toHaveAttribute("minlength");
      await user.click(passwordInput);
      await user.paste(password);
      await user.click(screen.getByRole("button", { name: "Sign in" }));

      await waitFor(() => expect(router.state.location.pathname).toBe("/"));
      expect(api.login).toHaveBeenCalledWith({
        username: "owner",
        password,
      });
    },
  );
});

it("returns to sign in when a domain request loses its session", async () => {
  const fetchImplementation = vi.fn(async (input: RequestInfo | URL) => {
    if ((input as Request).url.endsWith("/v1/auth/session"))
      return response(session);
    return new Response(
      JSON.stringify({
        code: "unauthorized",
        message: "Authentication is required",
        retryable: false,
      }),
      {
        status: 401,
        headers: {
          "content-type": "application/json",
          "X-Contractor-API-Version": "contractor.public.v1",
        },
      },
    );
  });
  const api = new PublicAPI(runtimeConfig, fetchImplementation);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: ["/runs"],
  });
  render(<Application api={api} publicAPI={api} router={router} />);
  expect(await screen.findByRole("region", { name: "Sign in" })).toBeVisible();
  expect(
    screen.queryByRole("heading", { name: "Runs" }),
  ).not.toBeInTheDocument();
  expect(api.csrf.get()).toBeUndefined();
  expect(
    fetchImplementation.mock.calls.filter(([input]) =>
      (input as Request).url.endsWith("/v1/auth/session"),
    ),
  ).toHaveLength(1);
});
