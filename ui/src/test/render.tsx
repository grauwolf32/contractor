// One render harness for unit tests: a memory router, a fresh query client,
// a PublicAPI over a fake Server and, when asked, a signed-in session.
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactElement } from "react";
import { createMemoryRouter, type RouteObject } from "react-router";
import { RouterProvider } from "react-router/dom";
import { vi } from "vitest";

import { PublicAPI, type AuthSession } from "../api/client";
import { PublicAPIProvider } from "../api/context";
import { queryKeys } from "../api/query-keys";
import { createApplicationQueryClient } from "../app/query-client";
import { SessionProvider, type SessionAPI } from "../auth/session";
import type { RuntimeConfig } from "../config/runtime-config";
import { fakeServer, type Fallback, type FakeServer } from "./fake-server";

export const TEST_RUNTIME_CONFIG: RuntimeConfig = {
  uiVersion: "0.1.0",
  supportedApiVersions: ["contractor.public.v1"],
  apiBaseUrl: "http://127.0.0.1:8080",
};

export const CSRF_TOKEN = "a".repeat(43);

type Capability = AuthSession["principal"]["capabilities"][number];

/** A signed-in owner session with `capabilities`. */
export function testSession(
  capabilities: Capability[] = ["user"],
): AuthSession {
  return {
    principal: { userId: "user_local", username: "owner", capabilities },
    csrfToken: CSRF_TOKEN,
    idleExpiresAt: "2026-10-06T20:00:00Z",
    absoluteExpiresAt: "2026-10-07T12:00:00Z",
  };
}

const servers = new WeakMap<PublicAPI, FakeServer>();

/**
 * A PublicAPI whose requests go to `server` (a FakeServer, or a handler
 * that becomes the fallback of an empty one). It holds a CSRF token, as
 * after sign-in.
 */
export function testAPI(server: FakeServer | Fallback): {
  api: PublicAPI;
  server: FakeServer;
} {
  const fake = typeof server === "function" ? fakeServer({}, server) : server;
  const api = new PublicAPI(TEST_RUNTIME_CONFIG, vi.fn(fake.fetch));
  api.csrf.replace(CSRF_TOKEN);
  servers.set(api, fake);
  return { api, server: fake };
}

/** The fake Server behind a PublicAPI from testAPI. */
export function serverOf(api: PublicAPI): FakeServer {
  const server = servers.get(api);
  if (server === undefined)
    throw new Error("The PublicAPI is not from testAPI");
  return server;
}

export interface RenderOptions {
  /** Where the router starts. */
  path?: string;
  /** Routes; without them `ui` renders at every path. */
  routes?: RouteObject[];
  /** Answers PublicAPI requests. */
  server?: FakeServer | Fallback;
  /** Signs in with this session (a SessionProvider over a stub API). */
  session?: AuthSession;
  queryClient?: QueryClient;
}

/**
 * Renders `ui` (or `routes`) with the application's providers. Paths that
 * match no route render "Elsewhere"; `location()` tells where navigation
 * went.
 */
export function renderWithAPI(
  ui: ReactElement | null,
  options: RenderOptions = {},
) {
  const { api, server } = testAPI(options.server ?? fakeServer());
  const queryClient = options.queryClient ?? createApplicationQueryClient();
  const routes = options.routes ?? [{ path: "*", element: ui }];
  const router = createMemoryRouter(
    [...routes, { path: "*", element: <p>Elsewhere</p> }],
    { initialEntries: [options.path ?? "/"] },
  );
  let tree = <RouterProvider router={router} />;
  let sessionAPI: SessionAPI | undefined;
  if (options.session !== undefined) {
    const session = options.session;
    queryClient.setQueryData(queryKeys.session, session);
    sessionAPI = {
      getSession: vi.fn(async () => session),
      login: vi.fn(async () => session),
      logout: vi.fn(async () => undefined),
    };
    tree = <SessionProvider api={sessionAPI}>{tree}</SessionProvider>;
  }
  const user = userEvent.setup();
  const view = render(
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={api}>{tree}</PublicAPIProvider>
    </QueryClientProvider>,
  );
  return {
    ...view,
    api,
    server,
    router,
    queryClient,
    sessionAPI,
    user,
    requests: server.requests,
    sent: server.sent,
    /** The current path and query string. */
    location: () =>
      router.state.location.pathname + router.state.location.search,
  };
}
