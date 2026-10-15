// The whole Application (session, shell, routes, live events) for route
// tests: a memory router at a path, a PublicAPI over a fake Server that signs
// the owner in, and an event socket the test opens and feeds frames.
import type { QueryClient } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import {
  createMemoryRouter,
  type InitialEntry,
  type RouteObject,
} from "react-router";

import type { AuthSession, PublicAPI } from "../api/client";
import { Application } from "../app/application";
import { applicationRoutes } from "../app/router";
import { RunEventsManager } from "../events/run-events";
import {
  emptyPageOrNotFound,
  fakeServer,
  json,
  type Fallback,
  type FakeServer,
  type Routes,
} from "./fake-server";
import { serverOf, testAPI, testSession, TEST_RUNTIME_CONFIG } from "./render";

type Capability = AuthSession["principal"]["capabilities"][number];

/** A WebSocket the test drives: it records frames and delivers messages. */
export class FakeEventSocket {
  /** The sockets of the current render, in the order they were opened. */
  static instances: FakeEventSocket[] = [];

  protocol = "contractor.events.v1";
  readyState = 0;
  sent: string[] = [];
  onopen: ((event: Event) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  onclose: ((event: CloseEvent) => unknown) | null = null;

  constructor(
    readonly url: string | URL,
    readonly protocols?: string | string[],
  ) {
    FakeEventSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  /** As a browser's: 1000 or an application code (3000–4999) only. */
  close(code?: number): void {
    if (
      code !== undefined &&
      code !== 1000 &&
      (!Number.isInteger(code) || code < 3000 || code > 4999)
    ) {
      throw new DOMException(
        "Invalid WebSocket close code",
        "InvalidAccessError",
      );
    }
    this.serverClose(code ?? 1000);
  }

  /** The Server closes the socket with `code`. */
  serverClose(code: number): void {
    this.readyState = 3;
    this.onclose?.({ code } as CloseEvent);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  message(value: unknown): void {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

export interface ApplicationOptions {
  /**
   * The fake Server: routes (other reads get an empty page where the
   * operation pages, else 404), a fallback handler, or a FakeServer.
   */
  server?: Routes | Fallback | FakeServer | undefined;
  /** A PublicAPI from testAPI, used instead of `server`. */
  api?: PublicAPI | undefined;
  /**
   * The signed-in owner's capabilities (default user and operations). The
   * harness routes GET /v1/auth/session unless the server routes it itself;
   * a session a fallback answers is not seen.
   */
  capabilities?: Capability[] | undefined;
  /** Entries before `path` in the history, so Back has somewhere to go. */
  history?: InitialEntry[] | undefined;
  /** Lazy routes, by route path, whose module loads once the promise settles. */
  slowRoutes?: Record<string, Promise<unknown>> | undefined;
  /** The query cache, for a test that seeds or watches it. */
  queryClient?: QueryClient | undefined;
  /** Renders in StrictMode. */
  strict?: boolean | undefined;
}

function isFakeServer(
  server: Routes | Fallback | FakeServer,
): server is FakeServer {
  return typeof server === "object" && "fetch" in server && "route" in server;
}

function fakeServerOf(options: ApplicationOptions): FakeServer {
  if (options.api !== undefined) return serverOf(options.api);
  const server = options.server ?? {};
  if (typeof server === "function") return fakeServer({}, server);
  if (isFakeServer(server)) return server;
  return fakeServer(server, emptyPageOrNotFound);
}

/** Makes the lazy route at `path` wait for `until`; false if there is none. */
function delayRoute(
  routes: RouteObject[],
  path: string,
  until: Promise<unknown>,
): boolean {
  for (const route of routes) {
    const { lazy } = route;
    if (route.path === path && typeof lazy === "function") {
      route.lazy = async () => {
        await until;
        return lazy();
      };
      return true;
    }
    if (route.children !== undefined && delayRoute(route.children, path, until))
      return true;
  }
  return false;
}

/**
 * Renders the Application at `path`, signed in, over a fake Server, with
 * live events delivered through FakeEventSocket (reset per call).
 */
export function renderApplication(
  path: InitialEntry = "/",
  options: ApplicationOptions = {},
) {
  FakeEventSocket.instances = [];
  const server = fakeServerOf(options);
  if (!server.handles("GET /v1/auth/session")) {
    const session = testSession(options.capabilities ?? ["user", "operations"]);
    server.route("GET /v1/auth/session", () => json(session));
  }
  const api = options.api ?? testAPI(server).api;
  const history = options.history ?? [];
  const routes = applicationRoutes();
  for (const [route, until] of Object.entries(options.slowRoutes ?? {})) {
    if (!delayRoute(routes, route, until))
      throw new Error(`No lazy route ${route}`);
  }
  const router = createMemoryRouter(routes, {
    initialEntries: [...history, path],
    initialIndex: history.length,
  });
  const events = new RunEventsManager(TEST_RUNTIME_CONFIG.apiBaseUrl, {
    WebSocketImplementation: FakeEventSocket as unknown as typeof WebSocket,
  });
  const application = (
    <Application
      api={api}
      publicAPI={api}
      runEvents={events}
      router={router}
      queryClient={options.queryClient}
    />
  );
  const view = render(
    options.strict === true ? (
      <StrictMode>{application}</StrictMode>
    ) : (
      application
    ),
  );
  return {
    ...view,
    api,
    server,
    router,
    events,
    user: userEvent.setup(),
    requests: server.requests,
    sent: server.sent,
    /** The current path and query string. */
    location: () =>
      router.state.location.pathname + router.state.location.search,
  };
}
