import type { PlannerPlan } from "../api/runs";
import { EventsSocket, type EventCursor } from "./socket";
import { conformsToEvent, eventFrameViolation } from "../api/contract";

const MAXIMUM_SERVER_FRAME_BYTES = 64 * 1024;

export type RunEventConnectionState =
  "connecting" | "live" | "reconnecting" | "resyncing" | "error";

export type RunResyncReason =
  | "generation_changed"
  | "cursor_unavailable"
  | "sequence_gap"
  | "protocol_error"
  | "projection_gap"
  | "subscription_error";

export interface PlannerActivity {
  kind:
    | "adk_event"
    | "scan_owner_acquired"
    | "scan_plan_initialized"
    | "scan_job_started"
    | "scan_job_finished";
  author:
    "user" | "streamline_planner" | "router_planner" | "other" | "scan_planner";
  functionCalls: string[];
  functionResults: string[];
  inputTokens?: number;
  outputTokens?: number;
  skipSummarization?: boolean;
  escalate?: boolean;
  truncated?: boolean;
}

export type PlannerEventKind =
  | "planner.started"
  | "planner.request_recorded"
  | "planner.activity"
  | "planner.plan_changed"
  | "planner.current_changed"
  | "planner.dispatch_selected"
  | "planner.dispatch_started"
  | "planner.dispatch_completed"
  | "planner.finish_requested"
  | "planner.completed"
  | "planner.failed";

export interface PlannerEventData {
  stageExecutionId: string;
  sessionId: string;
  invocationId: string;
  eventKind: PlannerEventKind;
  plan?: PlannerPlan;
  planRevision?: number;
  subtaskId?: string;
  callId?: string;
  workerName?: string;
  outcome?: "succeeded" | "failed" | "failure";
  code?: string;
  activity?: PlannerActivity;
}

export interface LifecycleEventData {
  runId: string;
  resource: "run" | "stageExecution";
  stageExecutionId?: string;
  state: string;
}

export interface OperationsEventData {
  resource:
    | "runtimeAgent"
    | "allocation"
    | "configuration"
    | "credential"
    | "schedulerSettings";
  resourceId?: string;
  revision: string;
}

export interface RunEventEnvelope<T> {
  cursor: EventCursor;
  occurredAt: string;
  data: T;
}

export interface RunEventCallbacks {
  onPlannerEvent: (event: RunEventEnvelope<PlannerEventData>) => boolean;
  onLifecycleEvent: (event: RunEventEnvelope<LifecycleEventData>) => void;
  onResync: (reason: RunResyncReason) => void;
  onStateChange: (state: RunEventConnectionState) => void;
  onError: (message: string) => void;
}

export interface OperationsEventCallbacks {
  onOperationsEvent: (event: RunEventEnvelope<OperationsEventData>) => void;
  onResync: (reason: RunResyncReason) => void;
  onStateChange: (state: RunEventConnectionState) => void;
  onError: (message: string) => void;
}

export interface RunEventSubscription {
  unsubscribe(): void;
}

export interface OperationsSnapshotCursor {
  generation: string;
  revision: string;
}

export interface OperationsEventSubscription {
  unsubscribe(): void;
  resume(after: OperationsSnapshotCursor): void;
}

type WebSocketFactory = new (
  url: string | URL,
  protocols?: string | string[],
) => WebSocket;

interface ManagerOptions {
  WebSocketImplementation?: WebSocketFactory;
  random?: () => number;
  schedule?: (
    callback: () => void,
    delay: number,
  ) => ReturnType<typeof setTimeout>;
  cancelSchedule?: (handle: ReturnType<typeof setTimeout>) => void;
}

interface RunSubscriptionRecord {
  kind: "run";
  id: string;
  runId: string;
  cursor: EventCursor;
  requestedCursor?: EventCursor;
  phase: "pending" | "subscribed" | "waiting" | "closing";
  callbacks: RunEventCallbacks;
}

interface OperationsSubscriptionRecord {
  kind: "operations";
  id: string;
  cursor: EventCursor;
  requestedCursor?: EventCursor;
  phase: "pending" | "subscribed" | "waiting" | "closing";
  callbacks: OperationsEventCallbacks;
}

type SubscriptionRecord = RunSubscriptionRecord | OperationsSubscriptionRecord;

type SubscribedFrame = {
  type: "subscribed";
  subscriptionId: string;
  stream: { kind: "run"; id: string } | { kind: "operations" };
  cursor: EventCursor;
};

type UnsubscribedFrame = {
  type: "unsubscribed";
  subscriptionId: string;
};

type RunEventFrame = {
  type: "event";
  subscriptionId: string;
  stream: { kind: "run"; id: string };
  cursor: EventCursor;
  occurredAt: string;
} & (
  | { kind: "planner.event"; data: PlannerEventData }
  | { kind: "lifecycle.changed"; data: LifecycleEventData }
);

type OperationsEventFrame = {
  type: "event";
  subscriptionId: string;
  stream: { kind: "operations" };
  cursor: EventCursor;
  occurredAt: string;
  kind: "operations.changed";
  data: OperationsEventData;
};

type ResyncFrame = {
  type: "resync_required";
  subscriptionId: string;
  stream: { kind: "run"; id: string } | { kind: "operations" };
  reason: "generation_changed" | "cursor_unavailable" | "sequence_gap";
};

type ErrorFrame = {
  type: "error";
  subscriptionId?: string;
  code:
    | "invalid_frame"
    | "subscription_limit"
    | "unauthorized"
    | "not_found"
    | "overloaded";
  message: string;
  retryable: boolean;
};

type ServerFrame =
  | SubscribedFrame
  | UnsubscribedFrame
  | RunEventFrame
  | OperationsEventFrame
  | ResyncFrame
  | ErrorFrame;

const SERVER_FRAME_TYPES = new Set([
  "subscribed",
  "unsubscribed",
  "event",
  "resync_required",
  "error",
]);

/**
 * Plan invariants contractor-events-v1 cannot express: subtask IDs are their
 * list positions, the current subtask is the first unresolved one, and a
 * running subtask exists exactly when an active dispatch selects it.
 */
function requirePlanInvariants(plan: PlannerPlan): void {
  if (plan.subtasks.some((subtask, index) => subtask.id !== String(index))) {
    throw new Error("Planner subtask ID order is invalid");
  }
  const firstUnresolved = plan.subtasks.find(
    (subtask) => subtask.status === "pending" || subtask.status === "running",
  );
  if (plan.currentSubtaskId !== firstUnresolved?.id) {
    throw new Error("Planner current subtask does not match its plan");
  }
  const running = plan.subtasks.filter(
    (subtask) => subtask.status === "running",
  );
  const dispatch = plan.activeDispatch;
  if (dispatch === undefined) {
    if (running.length !== 0) {
      throw new Error("Planner running subtask has no active dispatch");
    }
  } else if (
    running.length !== 1 ||
    plan.currentSubtaskId !== dispatch.subtaskId ||
    running[0]?.id !== dispatch.subtaskId
  ) {
    throw new Error("Planner active dispatch does not match its plan");
  }
}

/**
 * Parses one Server frame. The generated contractor-events-v1 schema checks
 * its shape; only invariants between members stay here.
 */
export function parseServerFrame(source: string): ServerFrame {
  if (
    source.length === 0 ||
    source.length > MAXIMUM_SERVER_FRAME_BYTES ||
    new TextEncoder().encode(source).byteLength > MAXIMUM_SERVER_FRAME_BYTES
  ) {
    throw new Error("Server event frame exceeds its bounded contract");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(source) as unknown;
  } catch {
    throw new Error("Server event frame is not valid JSON");
  }
  const violation = eventFrameViolation(parsed);
  if (violation !== undefined) {
    throw new Error(`Server event frame breaks the contract at ${violation}`);
  }
  const frame = parsed as ServerFrame | { type: "subscribe" | "unsubscribe" };
  if (!SERVER_FRAME_TYPES.has(frame.type)) {
    throw new Error("Server event frame type is a client frame");
  }
  const server = frame as ServerFrame;
  if (server.type === "event" && server.kind === "planner.event") {
    const data = server.data;
    if (data.plan !== undefined) requirePlanInvariants(data.plan);
    // A started dispatch is the plan's active dispatch.
    const active = data.plan?.activeDispatch;
    if (
      data.eventKind === "planner.dispatch_started" &&
      (active?.callId !== data.callId ||
        active?.subtaskId !== data.subtaskId ||
        active?.workerName !== data.workerName)
    ) {
      throw new Error("Started dispatch and Planner plan disagree");
    }
  }
  return server;
}

function sameCursor(left: EventCursor, right: EventCursor): boolean {
  return (
    left.generation === right.generation && left.sequence === right.sequence
  );
}

function copyCursor(cursor: EventCursor): EventCursor {
  return { generation: cursor.generation, sequence: cursor.sequence };
}

function validateCursor(cursor: EventCursor): void {
  if (!conformsToEvent("Cursor", cursor)) {
    throw new TypeError("Event cursor is invalid");
  }
}

function subscriptionMatchesStream(
  record: SubscriptionRecord,
  stream: { kind: "run"; id: string } | { kind: "operations" },
): boolean {
  return record.kind === "run"
    ? stream.kind === "run" && stream.id === record.runId
    : stream.kind === "operations";
}

export class RunEventsManager {
  readonly #apiBaseUrl: string;
  readonly #webSocketImplementation: WebSocketFactory | undefined;
  readonly #random: () => number;
  readonly #schedule: ManagerOptions["schedule"];
  readonly #cancelSchedule: ManagerOptions["cancelSchedule"];
  readonly #subscriptions = new Map<string, SubscriptionRecord>();
  #connection: EventsSocket | undefined;
  #reconnectHandle: ReturnType<typeof setTimeout> | undefined;
  #reconnectAttempt = 0;
  #nextSubscription = 1;

  constructor(apiBaseUrl: string, options: ManagerOptions = {}) {
    this.#apiBaseUrl = apiBaseUrl;
    this.#webSocketImplementation =
      options.WebSocketImplementation ?? globalThis.WebSocket;
    this.#random = options.random ?? Math.random;
    this.#schedule =
      options.schedule ??
      ((callback, delay) => globalThis.setTimeout(callback, delay));
    this.#cancelSchedule =
      options.cancelSchedule ?? ((handle) => globalThis.clearTimeout(handle));
  }

  subscribeRun(
    runId: string,
    after: EventCursor,
    callbacks: RunEventCallbacks,
  ): RunEventSubscription {
    if (!conformsToEvent("RunStream", { kind: "run", id: runId })) {
      throw new TypeError("Run ID is invalid");
    }
    validateCursor(after);
    const id = this.#subscriptionId("run");
    const record: RunSubscriptionRecord = {
      kind: "run",
      id,
      runId,
      cursor: copyCursor(after),
      phase: "pending",
      callbacks,
    };
    this.#subscriptions.set(id, record);
    callbacks.onStateChange("connecting");
    this.#ensureConnection();
    if (this.#connection?.socket.readyState === 1) {
      this.#sendSubscription(record);
    }
    let active = true;
    return {
      unsubscribe: () => {
        if (!active) {
          return;
        }
        active = false;
        this.#unsubscribe(record);
      },
    };
  }

  subscribeOperations(
    after: OperationsSnapshotCursor,
    callbacks: OperationsEventCallbacks,
  ): OperationsEventSubscription {
    const initialCursor = {
      generation: after.generation,
      sequence: after.revision,
    };
    validateCursor(initialCursor);
    const id = this.#subscriptionId("operations");
    const record: OperationsSubscriptionRecord = {
      kind: "operations",
      id,
      cursor: initialCursor,
      phase: "pending",
      callbacks,
    };
    this.#subscriptions.set(id, record);
    callbacks.onStateChange("connecting");
    this.#ensureConnection();
    if (this.#connection?.socket.readyState === 1) {
      this.#sendSubscription(record);
    }
    let active = true;
    return {
      unsubscribe: () => {
        if (!active) {
          return;
        }
        active = false;
        this.#unsubscribe(record);
      },
      resume: (after) => {
        if (
          !active ||
          record.phase !== "waiting" ||
          this.#subscriptions.get(record.id) !== record
        )
          return;
        const cursor = {
          generation: after.generation,
          sequence: after.revision,
        };
        validateCursor(cursor);
        record.cursor = cursor;
        record.phase = "pending";
        callbacks.onStateChange("connecting");
        this.#ensureConnection();
        if (this.#connection?.socket.readyState === 1) {
          this.#sendSubscription(record);
        }
      },
    };
  }

  #subscriptionId(kind: SubscriptionRecord["kind"]): string {
    const id = `${kind}-ui-${this.#nextSubscription}`;
    this.#nextSubscription += 1;
    return id;
  }

  close(): void {
    this.#cancelReconnect();
    const connection = this.#connection;
    this.#connection = undefined;
    connection?.close();
    this.#subscriptions.clear();
  }

  #ensureConnection(): void {
    if (
      this.#connection !== undefined ||
      this.#reconnectHandle !== undefined ||
      ![...this.#subscriptions.values()].some(
        (record) => record.phase === "pending",
      )
    ) {
      return;
    }
    if (this.#webSocketImplementation === undefined) {
      for (const record of this.#subscriptions.values()) {
        if (record.phase === "pending") {
          record.phase = "waiting";
          record.callbacks.onStateChange("error");
          record.callbacks.onError("WebSocket is unavailable in this browser");
        }
      }
      return;
    }
    let connection: EventsSocket;
    try {
      connection = new EventsSocket(
        this.#apiBaseUrl,
        this.#webSocketImplementation,
      );
    } catch {
      this.#scheduleReconnect();
      return;
    }
    this.#connection = connection;
    const socket = connection.socket;
    socket.onopen = () => this.#opened(connection);
    socket.onmessage = (event) => this.#message(connection, event);
    socket.onerror = () => {
      // The close frame is the reconnect authority. Browser error events carry
      // no safe diagnostic contract and are deliberately not logged.
    };
    socket.onclose = (event) => this.#closed(connection, event.code);
  }

  #opened(connection: EventsSocket): void {
    if (this.#connection !== connection) {
      connection.close();
      return;
    }
    try {
      connection.assertNegotiatedProtocol();
      for (const record of this.#subscriptions.values()) {
        if (record.phase === "pending") {
          this.#sendSubscription(record);
        }
      }
    } catch {
      this.#resyncAll("protocol_error");
    }
  }

  #sendSubscription(record: SubscriptionRecord): void {
    if (this.#connection === undefined || record.phase !== "pending") {
      return;
    }
    try {
      record.requestedCursor = copyCursor(record.cursor);
      this.#connection.subscribe(
        record.id,
        record.kind === "run"
          ? { kind: "run", id: record.runId }
          : { kind: "operations" },
        record.requestedCursor,
      );
    } catch {
      this.#resyncAll("protocol_error");
    }
  }

  #message(connection: EventsSocket, event: MessageEvent): void {
    if (this.#connection !== connection) {
      return;
    }
    if (typeof event.data !== "string") {
      this.#resyncAll("protocol_error");
      return;
    }
    let frame: ServerFrame;
    try {
      frame = parseServerFrame(event.data);
    } catch {
      this.#resyncAll("protocol_error");
      return;
    }
    switch (frame.type) {
      case "subscribed":
        this.#subscribed(frame);
        break;
      case "unsubscribed":
        this.#unsubscribed(frame);
        break;
      case "event":
        this.#event(frame);
        break;
      case "resync_required":
        this.#resyncFrame(frame);
        break;
      case "error":
        this.#errorFrame(frame);
        break;
    }
  }

  #subscribed(frame: SubscribedFrame): void {
    const record = this.#subscriptions.get(frame.subscriptionId);
    if (
      record === undefined ||
      (record.phase !== "pending" && record.phase !== "closing") ||
      record.requestedCursor === undefined ||
      !subscriptionMatchesStream(record, frame.stream) ||
      !sameCursor(frame.cursor, record.requestedCursor)
    ) {
      this.#resyncAll("protocol_error");
      return;
    }
    record.cursor = copyCursor(frame.cursor);
    delete record.requestedCursor;
    // The server may acknowledge subscribe after cancellation was sent.
    // Keep the record until unsubscribed so queued events remain harmless.
    if (record.phase === "closing") {
      return;
    }
    record.phase = "subscribed";
    this.#reconnectAttempt = 0;
    record.callbacks.onStateChange("live");
  }

  #unsubscribed(frame: UnsubscribedFrame): void {
    const record = this.#subscriptions.get(frame.subscriptionId);
    if (record === undefined || record.phase !== "closing") {
      this.#resyncAll("protocol_error");
      return;
    }
    this.#subscriptions.delete(record.id);
    this.#closeIfIdle();
  }

  #event(frame: RunEventFrame | OperationsEventFrame): void {
    const record = this.#subscriptions.get(frame.subscriptionId);
    if (record?.phase === "closing") {
      return;
    }
    if (
      record === undefined ||
      record.phase !== "subscribed" ||
      !subscriptionMatchesStream(record, frame.stream)
    ) {
      this.#resyncAll("protocol_error");
      return;
    }
    if (frame.cursor.generation !== record.cursor.generation) {
      this.#resyncSubscription(record, "generation_changed");
      return;
    }
    const current = BigInt(frame.cursor.sequence);
    const previous = BigInt(record.cursor.sequence);
    if (current <= previous) {
      return;
    }
    if (current !== previous + 1n) {
      this.#resyncSubscription(record, "sequence_gap");
      return;
    }
    try {
      if (record.kind === "operations") {
        if (
          frame.kind !== "operations.changed" ||
          frame.data.revision !== frame.cursor.sequence
        ) {
          this.#resyncAll("protocol_error");
          return;
        }
        record.callbacks.onOperationsEvent({
          cursor: copyCursor(frame.cursor),
          occurredAt: frame.occurredAt,
          data: frame.data,
        });
      } else if (frame.kind === "planner.event") {
        const accepted = record.callbacks.onPlannerEvent({
          cursor: copyCursor(frame.cursor),
          occurredAt: frame.occurredAt,
          data: frame.data,
        });
        if (!accepted) {
          this.#resyncSubscription(record, "projection_gap");
          return;
        }
      } else if (frame.kind === "lifecycle.changed") {
        if (frame.data.runId !== record.runId) {
          this.#resyncAll("protocol_error");
          return;
        }
        record.callbacks.onLifecycleEvent({
          cursor: copyCursor(frame.cursor),
          occurredAt: frame.occurredAt,
          data: frame.data,
        });
      } else {
        this.#resyncAll("protocol_error");
        return;
      }
    } catch {
      this.#resyncSubscription(record, "projection_gap");
      return;
    }
    record.cursor = copyCursor(frame.cursor);
  }

  #resyncFrame(frame: ResyncFrame): void {
    const record = this.#subscriptions.get(frame.subscriptionId);
    if (
      record === undefined ||
      !subscriptionMatchesStream(record, frame.stream)
    ) {
      this.#resyncAll("protocol_error");
      return;
    }
    if (record.phase === "closing") {
      return;
    }
    this.#resyncSubscription(record, frame.reason);
  }

  #errorFrame(frame: ErrorFrame): void {
    if (frame.subscriptionId !== undefined) {
      const record = this.#subscriptions.get(frame.subscriptionId);
      // A failed subscribe and its racing unsubscribe can each report
      // not_found. The second reply has no remaining subscription to affect.
      if (record === undefined && frame.code === "not_found") {
        return;
      }
      if (record?.phase === "closing" && frame.code === "not_found") {
        this.#subscriptions.delete(record.id);
        this.#closeIfIdle();
        return;
      }
      if (record?.phase === "closing") {
        return;
      }
      if (record !== undefined) {
        record.callbacks.onError(frame.message);
        this.#resyncSubscription(record, "subscription_error");
        return;
      }
    } else {
      for (const record of this.#subscriptions.values()) {
        record.callbacks.onError(frame.message);
      }
    }
    this.#resyncAll("subscription_error");
  }

  #closed(connection: EventsSocket, code: number): void {
    if (this.#connection !== connection) {
      return;
    }
    this.#connection = undefined;
    const active = [...this.#subscriptions.values()].filter(
      (record) => record.phase !== "closing" && record.phase !== "waiting",
    );
    for (const record of [...this.#subscriptions.values()]) {
      if (record.phase === "closing") {
        this.#subscriptions.delete(record.id);
      } else if (record.phase !== "waiting") {
        record.phase = "pending";
        delete record.requestedCursor;
      }
    }
    if (code === 1008) {
      for (const record of active) {
        record.phase = "waiting";
        record.callbacks.onStateChange("error");
        record.callbacks.onError("Live event session is no longer authorized");
      }
      return;
    }
    for (const record of active) {
      record.callbacks.onStateChange("reconnecting");
    }
    this.#scheduleReconnect();
  }

  #scheduleReconnect(): void {
    if (
      this.#reconnectHandle !== undefined ||
      ![...this.#subscriptions.values()].some(
        (record) => record.phase === "pending",
      )
    ) {
      return;
    }
    const maximum = Math.min(30_000, 500 * 2 ** this.#reconnectAttempt);
    this.#reconnectAttempt += 1;
    const delay = Math.floor(
      Math.max(0, Math.min(1, this.#random())) * maximum,
    );
    this.#reconnectHandle = this.#schedule?.(() => {
      this.#reconnectHandle = undefined;
      this.#ensureConnection();
    }, delay);
  }

  #cancelReconnect(): void {
    if (this.#reconnectHandle !== undefined) {
      this.#cancelSchedule?.(this.#reconnectHandle);
      this.#reconnectHandle = undefined;
    }
  }

  /**
   * Sends one subscription back to its owner for an authoritative baseline
   * while the socket and every other subscription stay live. Protocol
   * violations still reset the whole socket through `#resyncAll`.
   */
  #resyncSubscription(
    record: SubscriptionRecord,
    reason: RunResyncReason,
  ): void {
    const connection = this.#connection;
    if (
      connection?.socket.readyState === 1 &&
      (record.phase === "subscribed" ||
        (record.phase === "pending" && record.requestedCursor !== undefined))
    ) {
      // The Server may still pump the old subscription, or not have released
      // its ID yet. Cancel it and keep a closing record under the old ID so
      // its late frames and the reply (unsubscribed or not_found) stay
      // harmless; the owner unsubscribes and subscribes again after its REST
      // baseline refetch.
      try {
        connection.unsubscribe(record.id);
      } catch {
        this.#resyncAll("protocol_error");
        return;
      }
      this.#subscriptions.set(record.id, { ...record, phase: "closing" });
    } else {
      this.#subscriptions.delete(record.id);
    }
    record.id = this.#subscriptionId(record.kind);
    record.phase = "waiting";
    delete record.requestedCursor;
    this.#subscriptions.set(record.id, record);
    record.callbacks.onStateChange("resyncing");
    record.callbacks.onResync(reason);
  }

  #resyncAll(reason: RunResyncReason): void {
    this.#cancelReconnect();
    const connection = this.#connection;
    this.#connection = undefined;
    if (connection !== undefined) {
      try {
        connection.socket.close(4002, "authoritative resync required");
      } catch {
        // A browser or test transport can reject close; retry with a normal
        // closure, then keep notifying owners even if the socket is unusable.
        try {
          connection.close();
        } catch {
          // The subscription reset below must not depend on socket teardown.
        }
      }
    }
    for (const record of this.#subscriptions.values()) {
      if (record.phase === "closing") {
        this.#subscriptions.delete(record.id);
        continue;
      }
      record.phase = "waiting";
      delete record.requestedCursor;
      record.callbacks.onStateChange("resyncing");
      record.callbacks.onResync(reason);
    }
  }

  #unsubscribe(record: SubscriptionRecord): void {
    if (!this.#subscriptions.has(record.id)) {
      return;
    }
    if (
      this.#connection?.socket.readyState === 1 &&
      (record.phase === "subscribed" ||
        (record.phase === "pending" && record.requestedCursor !== undefined))
    ) {
      record.phase = "closing";
      try {
        this.#connection.unsubscribe(record.id);
      } catch {
        this.#subscriptions.delete(record.id);
        this.#resyncAll("protocol_error");
      }
      return;
    }
    this.#subscriptions.delete(record.id);
    this.#closeIfIdle();
  }

  #closeIfIdle(): void {
    if (this.#subscriptions.size !== 0) {
      return;
    }
    this.#cancelReconnect();
    const connection = this.#connection;
    this.#connection = undefined;
    connection?.close();
  }
}
