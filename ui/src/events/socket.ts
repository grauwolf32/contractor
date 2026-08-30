import { validateAPIBaseURL } from "../config/runtime-config";

export const EVENT_PROTOCOL = "contractor.events.v1";

const SAFE_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
const UNSIGNED_DECIMAL = /^(?:0|[1-9][0-9]*)$/;

export interface EventCursor {
  generation: string;
  sequence: string;
}

export type EventStream = { kind: "run"; id: string } | { kind: "operations" };

type WebSocketFactory = new (
  url: string | URL,
  protocols?: string | string[],
) => WebSocket;

function requireIdentifier(name: string, value: string, maximum = 256): void {
  if (value.length > maximum || !SAFE_IDENTIFIER.test(value)) {
    throw new TypeError(`${name} is not a safe identifier`);
  }
}

function requireCursor(cursor: EventCursor): void {
  requireIdentifier("cursor generation", cursor.generation);
  if (cursor.sequence.length > 20 || !UNSIGNED_DECIMAL.test(cursor.sequence)) {
    throw new TypeError("cursor sequence is not an unsigned decimal string");
  }
}

export function eventSocketURL(apiBaseUrl: string): string {
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
      this.socket.close(1002, "subprotocol mismatch");
      throw new Error("Server did not negotiate contractor.events.v1");
    }
  }

  subscribe(
    subscriptionId: string,
    stream: EventStream,
    after?: EventCursor,
  ): void {
    requireIdentifier("subscription ID", subscriptionId, 128);
    if (stream.kind === "run") {
      requireIdentifier("Run ID", stream.id);
    }
    if (after !== undefined) {
      requireCursor(after);
    }
    const exactStream =
      stream.kind === "run"
        ? { kind: "run" as const, id: stream.id }
        : { kind: "operations" as const };
    this.socket.send(
      JSON.stringify({
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
      }),
    );
  }

  unsubscribe(subscriptionId: string): void {
    requireIdentifier("subscription ID", subscriptionId, 128);
    this.socket.send(
      JSON.stringify({
        version: EVENT_PROTOCOL,
        type: "unsubscribe",
        subscriptionId,
      }),
    );
  }

  close(): void {
    this.socket.close(1000, "client closed");
  }
}
