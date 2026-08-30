import { describe, expect, it } from "vitest";

import { EVENT_PROTOCOL, EventsSocket, eventSocketURL } from "./socket";

class FakeWebSocket {
  readonly url: string;
  readonly requestedProtocol: string | string[] | undefined;
  protocol = EVENT_PROTOCOL;
  sent: string[] = [];
  closed: [number | undefined, string | undefined] | undefined;

  constructor(url: string | URL, protocols?: string | string[]) {
    this.url = String(url);
    this.requestedProtocol = protocols;
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(code?: number, reason?: string): void {
    this.closed = [code, reason];
  }
}

describe("EventsSocket", () => {
  it("derives a secret-free direct WebSocket URL", () => {
    expect(eventSocketURL("https://api.example.test:8443")).toBe(
      "wss://api.example.test:8443/v1/events/ws",
    );
    expect(eventSocketURL("http://127.0.0.1:8080")).toBe(
      "ws://127.0.0.1:8080/v1/events/ws",
    );
    expect(() => eventSocketURL("http://localhost:8080")).toThrow();
  });

  it("sends bounded protocol frames without URL credentials or cursors", () => {
    const events = new EventsSocket(
      "http://127.0.0.1:8080",
      FakeWebSocket as unknown as typeof WebSocket,
    );
    const socket = events.socket as unknown as FakeWebSocket;
    expect(socket.url).toBe("ws://127.0.0.1:8080/v1/events/ws");
    expect(socket.requestedProtocol).toBe(EVENT_PROTOCOL);

    events.assertNegotiatedProtocol();
    events.subscribe(
      "run-detail",
      { kind: "run", id: "run_123" },
      { generation: "run-generation-123", sequence: "42" },
    );
    events.unsubscribe("run-detail");
    expect(JSON.parse(socket.sent[0] ?? "{}")).toEqual({
      version: EVENT_PROTOCOL,
      type: "subscribe",
      subscriptionId: "run-detail",
      stream: { kind: "run", id: "run_123" },
      after: { generation: "run-generation-123", sequence: "42" },
    });
    expect(JSON.parse(socket.sent[1] ?? "{}")).toEqual({
      version: EVENT_PROTOCOL,
      type: "unsubscribe",
      subscriptionId: "run-detail",
    });
  });

  it("closes a connection that negotiated another subprotocol", () => {
    const events = new EventsSocket(
      "http://127.0.0.1:8080",
      FakeWebSocket as unknown as typeof WebSocket,
    );
    const socket = events.socket as unknown as FakeWebSocket;
    socket.protocol = "other";
    expect(() => events.assertNegotiatedProtocol()).toThrow("negotiate");
    expect(socket.closed).toEqual([1002, "subprotocol mismatch"]);
  });
});
