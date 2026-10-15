// Test support for the Runs area (the Run page and the Queue): the whole
// application over the shared fake Server, with a fake event socket the
// test opens and feeds frames.
//
// V385: needs shared application harness with a fake event socket in
// src/test; until then the Runs and Queue tests share this one.
import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter } from "react-router";

import { Application } from "../../app/application";
import { applicationRoutes } from "../../app/router";
import { RunEventsManager } from "../../events/run-events";
import {
  json,
  type Fallback,
  type FakeServer,
  type Routes,
} from "../../test/fake-server";
import { testAPI, testSession, TEST_RUNTIME_CONFIG } from "../../test/render";

/** A WebSocket the test drives: it records frames and delivers messages. */
export class FakeEventSocket {
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

/** The signed-in owner, with Operations access. */
export const SIGNED_IN: Routes = {
  "GET /v1/auth/session": () => json(testSession(["user", "operations"])),
};

/**
 * The application at `path` over `server`, with run events delivered
 * through FakeEventSocket (reset per call).
 */
export function renderApplication(server: FakeServer | Fallback, path: string) {
  FakeEventSocket.instances = [];
  const { api, server: fake } = testAPI(server);
  const router = createMemoryRouter(applicationRoutes(), {
    initialEntries: [path],
  });
  const events = new RunEventsManager(TEST_RUNTIME_CONFIG.apiBaseUrl, {
    WebSocketImplementation: FakeEventSocket as unknown as typeof WebSocket,
  });
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
    server: fake,
    router,
    events,
    user: userEvent.setup(),
    requests: fake.requests,
    sent: fake.sent,
    /** The current path and query string. */
    location: () =>
      router.state.location.pathname + router.state.location.search,
  };
}
