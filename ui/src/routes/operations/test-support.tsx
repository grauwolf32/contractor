// Test harness of the Operations and Settings pages: the whole Application
// (shell, session, live events) over the shared fake Server, with a
// WebSocket the test drives.
// V385: needs shared Application harness in src/test (queue.test.tsx and
// runs/runs.test.tsx build the same one).
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";

import type { AuthSession } from "../../api/client";
import type { OperationsSnapshot } from "../../api/operations";
import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import { RunEventsManager } from "../../events/run-events";
import { conforming } from "../../test/builders";
import {
  fakeServer,
  json,
  type Fallback,
  type Routes,
} from "../../test/fake-server";
import { testAPI, testSession, TEST_RUNTIME_CONFIG } from "../../test/render";

type Capability = AuthSession["principal"]["capabilities"][number];

/** An event socket that stays connecting until the test opens it. */
export class FakeSocket {
  static instances: FakeSocket[] = [];
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
    FakeSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(code?: number): void {
    this.readyState = 3;
    this.onclose?.({ code: code ?? 1000 } as CloseEvent);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  message(value: unknown): void {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent);
  }
}

/** An empty Operations snapshot at `revision`. */
export function operationsSnapshot(
  revision = "7",
  overrides: Partial<OperationsSnapshot> = {},
): OperationsSnapshot {
  return conforming("OperationsSnapshot", {
    cursor: { generation: "operations-generation-1", revision },
    runtimeAgents: [],
    allocations: [],
    ...overrides,
  });
}

export interface OperationsAppOptions {
  /** Routes of the fake Server; the session route is added unless given. */
  routes?: Routes;
  fallback?: Fallback;
  /** The signed-in session's capabilities; default user and operations. */
  capabilities?: Capability[];
}

/**
 * Renders the Application at `path` against a fake Server that answers the
 * session with `capabilities` and `routes`. FakeSocket.instances is reset.
 */
export function renderOperationsApp(
  path: string,
  options: OperationsAppOptions = {},
) {
  FakeSocket.instances = [];
  const session = testSession(options.capabilities ?? ["user", "operations"]);
  const server = fakeServer(
    { "GET /v1/auth/session": () => json(session), ...options.routes },
    options.fallback,
  );
  const { api } = testAPI(server);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const events = new RunEventsManager(TEST_RUNTIME_CONFIG.apiBaseUrl, {
    WebSocketImplementation: FakeSocket as unknown as typeof WebSocket,
  });
  const user = userEvent.setup();
  const view = render(
    <Application
      api={api}
      publicAPI={api}
      runEvents={events}
      router={router}
    />,
  );
  return {
    ...view,
    api,
    server,
    router,
    user,
    requests: server.requests,
    sent: server.sent,
    /** The current path and query string. */
    location: () =>
      router.state.location.pathname + router.state.location.search,
  };
}
