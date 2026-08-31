import { describe, expect, it, vi } from "vitest";

import {
  parseServerFrame,
  RunEventsManager,
  type OperationsEventCallbacks,
  type RunEventCallbacks,
  type RunEventConnectionState,
  type RunResyncReason,
} from "./run-events";

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readonly requestedProtocol: string | string[] | undefined;
  protocol = "contractor.events.v1";
  readyState = 0;
  sent: string[] = [];
  closed: [number | undefined, string | undefined] | undefined;
  onopen: ((event: Event) => unknown) | null = null;
  onmessage: ((event: MessageEvent) => unknown) | null = null;
  onerror: ((event: Event) => unknown) | null = null;
  onclose: ((event: CloseEvent) => unknown) | null = null;

  constructor(url: string | URL, protocols?: string | string[]) {
    this.url = String(url);
    this.requestedProtocol = protocols;
    FakeWebSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(code?: number, reason?: string): void {
    this.closed = [code, reason];
    this.readyState = 3;
    this.onclose?.({ code: code ?? 1000 } as CloseEvent);
  }

  open(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  message(value: unknown): void {
    const data = typeof value === "string" ? value : JSON.stringify(value);
    this.onmessage?.({ data } as MessageEvent);
  }

  serverClose(code = 1006): void {
    this.readyState = 3;
    this.onclose?.({ code } as CloseEvent);
  }
}

function cursor(sequence: string) {
  return { generation: "run-generation-1", sequence };
}

function subscribed(subscriptionId: string, runId: string, sequence: string) {
  return {
    version: "contractor.events.v1",
    type: "subscribed",
    subscriptionId,
    stream: { kind: "run", id: runId },
    cursor: cursor(sequence),
  };
}

function plannerEvent(subscriptionId: string, runId: string, sequence: string) {
  return {
    version: "contractor.events.v1",
    type: "event",
    subscriptionId,
    stream: { kind: "run", id: runId },
    cursor: cursor(sequence),
    kind: "planner.event",
    occurredAt: "2026-08-31T12:00:00Z",
    data: {
      stageExecutionId: "stage-1",
      sessionId: "session-1",
      invocationId: "invocation-1",
      eventKind: "planner.plan_changed",
      plan: {
        revision: 1,
        subtasks: [
          {
            id: "0",
            objective: "Inspect source",
            instructions: "Use the bounded source tools",
            status: "pending",
          },
        ],
        currentSubtaskId: "0",
      },
    },
  };
}

function lifecycleEvent(
  subscriptionId: string,
  runId: string,
  sequence: string,
) {
  return {
    version: "contractor.events.v1",
    type: "event",
    subscriptionId,
    stream: { kind: "run", id: runId },
    cursor: cursor(sequence),
    kind: "lifecycle.changed",
    occurredAt: "2026-08-31T12:00:01Z",
    data: { runId, resource: "run", state: "running" },
  };
}

function callbacks() {
  const states: RunEventConnectionState[] = [];
  const resyncs: RunResyncReason[] = [];
  const planner = vi.fn(() => true);
  const lifecycle = vi.fn();
  const errors: string[] = [];
  const value: RunEventCallbacks = {
    onPlannerEvent: planner,
    onLifecycleEvent: lifecycle,
    onResync: (reason) => resyncs.push(reason),
    onStateChange: (state) => states.push(state),
    onError: (message) => errors.push(message),
  };
  return { value, states, resyncs, planner, lifecycle, errors };
}

function operationsCallbacks() {
  const states: RunEventConnectionState[] = [];
  const resyncs: RunResyncReason[] = [];
  const changed = vi.fn();
  const errors: string[] = [];
  const value: OperationsEventCallbacks = {
    onOperationsEvent: changed,
    onResync: (reason) => resyncs.push(reason),
    onStateChange: (state) => states.push(state),
    onError: (message) => errors.push(message),
  };
  return { value, states, resyncs, changed, errors };
}

describe("Run event protocol", () => {
  it("parses only the reduced typed Planner projection", () => {
    const parsed = parseServerFrame(
      JSON.stringify(plannerEvent("run-ui-1", "run-1", "42")),
    );
    expect(parsed.type).toBe("event");
    if (parsed.type !== "event" || parsed.kind !== "planner.event") {
      throw new Error("expected Planner event");
    }
    expect(parsed.data.plan?.subtasks[0]?.objective).toBe("Inspect source");
    expect("prompt" in parsed.data).toBe(false);
  });

  it("rejects unknown fields instead of admitting prompt or tool payload data", () => {
    const unsafe = plannerEvent("run-ui-1", "run-1", "42");
    const data = unsafe.data as typeof unsafe.data & {
      prompt?: string;
      toolArguments?: unknown;
    };
    data.prompt = "secret prompt";
    data.toolArguments = { token: "secret" };
    expect(() => parseServerFrame(JSON.stringify(unsafe))).toThrow(
      "closed shape",
    );
  });

  it("parses a closed Operations invalidation without resource payload", () => {
    const parsed = parseServerFrame(
      JSON.stringify({
        version: "contractor.events.v1",
        type: "event",
        subscriptionId: "operations-ui-1",
        stream: { kind: "operations" },
        cursor: { generation: "operations-generation-1", sequence: "10" },
        kind: "operations.changed",
        occurredAt: "2026-08-31T12:00:00Z",
        data: {
          resource: "allocation",
          resourceId: "allocation-1",
          revision: "10",
        },
      }),
    );
    expect(parsed).toMatchObject({
      type: "event",
      kind: "operations.changed",
      data: { resource: "allocation", revision: "10" },
    });
    expect(JSON.stringify(parsed)).not.toMatch(/observedState|credentialId/);
  });
});

describe("RunEventsManager", () => {
  it("multiplexes subscriptions and accepts only contiguous cursor events", () => {
    FakeWebSocket.instances = [];
    const first = callbacks();
    const second = callbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    manager.subscribeRun("run-1", cursor("41"), first.value);
    manager.subscribeRun("run-2", cursor("8"), second.value);
    expect(FakeWebSocket.instances).toHaveLength(1);
    const socket = FakeWebSocket.instances[0];
    socket?.open();
    expect(socket?.sent.map((frame) => JSON.parse(frame))).toEqual([
      {
        version: "contractor.events.v1",
        type: "subscribe",
        subscriptionId: "run-ui-1",
        stream: { kind: "run", id: "run-1" },
        after: cursor("41"),
      },
      {
        version: "contractor.events.v1",
        type: "subscribe",
        subscriptionId: "run-ui-2",
        stream: { kind: "run", id: "run-2" },
        after: cursor("8"),
      },
    ]);
    socket?.message(subscribed("run-ui-1", "run-1", "41"));
    socket?.message(subscribed("run-ui-2", "run-2", "8"));
    socket?.message(plannerEvent("run-ui-1", "run-1", "42"));
    socket?.message(plannerEvent("run-ui-1", "run-1", "42"));
    socket?.message(lifecycleEvent("run-ui-1", "run-1", "43"));
    expect(first.planner).toHaveBeenCalledTimes(1);
    expect(first.lifecycle).toHaveBeenCalledTimes(1);
    expect(first.states).toEqual(["connecting", "live"]);

    socket?.message(lifecycleEvent("run-ui-1", "run-1", "45"));
    expect(first.resyncs).toEqual(["sequence_gap"]);
    expect(second.resyncs).toEqual(["sequence_gap"]);
    expect(socket?.closed?.[0]).toBe(1002);
  });

  it("adds a second Run subscription to an already-open socket", () => {
    FakeWebSocket.instances = [];
    const first = callbacks();
    const second = callbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    manager.subscribeRun("run-1", cursor("4"), first.value);
    const socket = FakeWebSocket.instances[0];
    socket?.open();
    socket?.message(subscribed("run-ui-1", "run-1", "4"));

    manager.subscribeRun("run-2", cursor("9"), second.value);

    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(JSON.parse(socket?.sent.at(-1) ?? "{}")).toEqual({
      version: "contractor.events.v1",
      type: "subscribe",
      subscriptionId: "run-ui-2",
      stream: { kind: "run", id: "run-2" },
      after: cursor("9"),
    });
    socket?.message(subscribed("run-ui-2", "run-2", "9"));
    expect(first.states).toEqual(["connecting", "live"]);
    expect(second.states).toEqual(["connecting", "live"]);
  });

  it("multiplexes Operations invalidation on the existing Run socket", () => {
    FakeWebSocket.instances = [];
    const run = callbacks();
    const operations = operationsCallbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    manager.subscribeRun("run-1", cursor("4"), run.value);
    const socket = FakeWebSocket.instances[0];
    socket?.open();
    socket?.message(subscribed("run-ui-1", "run-1", "4"));

    manager.subscribeOperations(
      { generation: "operations-generation-1", revision: "9" },
      operations.value,
    );
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(JSON.parse(socket?.sent.at(-1) ?? "{}")).toEqual({
      version: "contractor.events.v1",
      type: "subscribe",
      subscriptionId: "operations-ui-2",
      stream: { kind: "operations" },
      after: { generation: "operations-generation-1", sequence: "9" },
    });
    socket?.message({
      version: "contractor.events.v1",
      type: "subscribed",
      subscriptionId: "operations-ui-2",
      stream: { kind: "operations" },
      cursor: { generation: "operations-generation-1", sequence: "9" },
    });
    const changed = {
      version: "contractor.events.v1",
      type: "event",
      subscriptionId: "operations-ui-2",
      stream: { kind: "operations" },
      cursor: { generation: "operations-generation-1", sequence: "10" },
      kind: "operations.changed",
      occurredAt: "2026-08-31T12:00:00Z",
      data: {
        resource: "runtimeAgent",
        resourceId: "runtime-1",
        revision: "10",
      },
    };
    socket?.message(changed);
    socket?.message(changed);
    expect(operations.changed).toHaveBeenCalledTimes(1);
    expect(operations.states).toEqual(["connecting", "live"]);
    expect(run.states).toEqual(["connecting", "live"]);
  });

  it("resynchronizes every projection on generation mismatch", () => {
    FakeWebSocket.instances = [];
    const current = callbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    manager.subscribeRun("run-1", cursor("4"), current.value);
    const socket = FakeWebSocket.instances[0];
    socket?.open();
    socket?.message(subscribed("run-ui-1", "run-1", "4"));
    const changedGeneration = plannerEvent("run-ui-1", "run-1", "5");
    changedGeneration.cursor = {
      generation: "run-generation-2",
      sequence: "5",
    };
    socket?.message(changedGeneration);

    expect(current.resyncs).toEqual(["generation_changed"]);
    expect(current.states).toEqual(["connecting", "live", "resyncing"]);
    expect(socket?.closed?.[0]).toBe(1002);
  });

  it("honors explicit resync frames and treats unknown frames as protocol gaps", () => {
    FakeWebSocket.instances = [];
    const explicit = callbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    manager.subscribeRun("run-1", cursor("4"), explicit.value);
    const first = FakeWebSocket.instances[0];
    first?.open();
    first?.message(subscribed("run-ui-1", "run-1", "4"));
    first?.message({
      version: "contractor.events.v1",
      type: "resync_required",
      subscriptionId: "run-ui-1",
      stream: { kind: "run", id: "run-1" },
      reason: "cursor_unavailable",
    });
    expect(explicit.resyncs).toEqual(["cursor_unavailable"]);

    const replacement = callbacks();
    const nextManager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    nextManager.subscribeRun("run-2", cursor("1"), replacement.value);
    const second = FakeWebSocket.instances[1];
    second?.open();
    second?.message(subscribed("run-ui-1", "run-2", "1"));
    second?.message({
      version: "contractor.events.v1",
      type: "future_frame",
      opaque: "must not be retained",
    });
    expect(replacement.resyncs).toEqual(["protocol_error"]);
    expect(replacement.errors).toEqual([]);
  });

  it("reconnects with full jitter and the last fully processed cursor", () => {
    FakeWebSocket.instances = [];
    const scheduled: Array<{ callback: () => void; delay: number }> = [];
    const current = callbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
      random: () => 0.5,
      schedule: (callback, delay) => {
        scheduled.push({ callback, delay });
        return scheduled.length as unknown as ReturnType<typeof setTimeout>;
      },
      cancelSchedule: vi.fn(),
    });
    manager.subscribeRun("run-1", cursor("1"), current.value);
    const first = FakeWebSocket.instances[0];
    first?.open();
    first?.message(subscribed("run-ui-1", "run-1", "1"));
    first?.message(plannerEvent("run-ui-1", "run-1", "2"));
    first?.serverClose(1013);
    expect(current.states).toEqual(["connecting", "live", "reconnecting"]);
    expect(scheduled[0]?.delay).toBe(250);

    scheduled[0]?.callback();
    const reconnected = FakeWebSocket.instances[1];
    reconnected?.open();
    expect(JSON.parse(reconnected?.sent[0] ?? "{}").after).toEqual(cursor("2"));
  });

  it("requires an authoritative baseline when a projection cannot advance", () => {
    FakeWebSocket.instances = [];
    const current = callbacks();
    current.value.onPlannerEvent = vi.fn(() => false);
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    const subscription = manager.subscribeRun(
      "run-1",
      cursor("4"),
      current.value,
    );
    const first = FakeWebSocket.instances[0];
    first?.open();
    first?.message(subscribed("run-ui-1", "run-1", "4"));
    first?.message(plannerEvent("run-ui-1", "run-1", "5"));
    expect(current.resyncs).toEqual(["projection_gap"]);

    subscription.resume(cursor("7"));
    const resumed = FakeWebSocket.instances[1];
    resumed?.open();
    expect(JSON.parse(resumed?.sent[0] ?? "{}").after).toEqual(cursor("7"));
  });

  it("waits for unsubscribe acknowledgement before closing the idle socket", () => {
    FakeWebSocket.instances = [];
    const current = callbacks();
    const manager = new RunEventsManager("http://127.0.0.1:8080", {
      WebSocketImplementation: FakeWebSocket as unknown as typeof WebSocket,
    });
    const subscription = manager.subscribeRun(
      "run-1",
      cursor("0"),
      current.value,
    );
    const socket = FakeWebSocket.instances[0];
    socket?.open();
    socket?.message(subscribed("run-ui-1", "run-1", "0"));
    subscription.unsubscribe();
    expect(JSON.parse(socket?.sent.at(-1) ?? "{}").type).toBe("unsubscribe");
    expect(socket?.closed).toBeUndefined();
    socket?.message({
      version: "contractor.events.v1",
      type: "unsubscribed",
      subscriptionId: "run-ui-1",
    });
    expect(socket?.closed).toEqual([1000, "client closed"]);
  });
});
