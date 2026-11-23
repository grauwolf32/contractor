import { eventFrameViolation } from "../api/contract";
import { validateAPIBaseURL } from "../config/runtime-config";

const EVENT_PROTOCOL = "contractor.events.v1";

export interface EventCursor {
  generation: string;
  sequence: string;
}

type EventStream = { kind: "run"; id: string } | { kind: "operations" };

type WebSocketFactory = new (
  url: string | URL,
  protocols?: string | string[],
) => WebSocket;

/** Sends a client frame after checking it against contractor-events-v1. */
function send(socket: WebSocket, frame: object): void {
  if (eventFrameViolation(frame) !== undefined) {
    throw new TypeError("Event client frame breaks the contract");
  }
  socket.send(JSON.stringify(frame));
}

function eventSocketURL(apiBaseUrl: string): string {
  const origin = validateAPIBaseURL(apiBaseUrl);
  const url = new URL("/v1/events/ws", origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.href;
}

export class EventsSocket {
  readonly socket: WebSocket;

  constructor(
    apiBaseUrl: string,
    WebSocketImplementation: WebSocketFactory = globalThis.WebSocket,
  ) {
    this.socket = new WebSocketImplementation(
      eventSocketURL(apiBaseUrl),
      EVENT_PROTOCOL,
    );
  }

  assertNegotiatedProtocol(): void {
    if (this.socket.protocol !== EVENT_PROTOCOL) {
      this.socket.close(4003, "subprotocol mismatch");
      throw new Error("Server did not negotiate contractor.events.v1");
    }
  }

  subscribe(
    subscriptionId: string,
    stream: EventStream,
    after?: EventCursor,
  ): void {
    const exactStream =
      stream.kind === "run"
        ? { kind: "run" as const, id: stream.id }
        : { kind: "operations" as const };
    send(this.socket, {
      version: EVENT_PROTOCOL,
      type: "subscribe",
      subscriptionId,
      stream: exactStream,
      ...(after === undefined
        ? {}
        : {
            after: {
              generation: after.generation,
              sequence: after.sequence,
            },
          }),
    });
  }

  unsubscribe(subscriptionId: string): void {
    send(this.socket, {
      version: EVENT_PROTOCOL,
      type: "unsubscribe",
      subscriptionId,
    });
  }

  close(): void {
    this.socket.close(1000, "client closed");
  }
}
