import type { PlannerPlan } from "../api/runs";
import { EVENT_PROTOCOL, EventsSocket, type EventCursor } from "./socket";

const MAXIMUM_SERVER_FRAME_BYTES = 64 * 1024;
const SAFE_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$/;
const SUBSCRIPTION_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;
const UNSIGNED_DECIMAL = /^(?:0|[1-9][0-9]*)$/;
const SUBTASK_IDENTIFIER = /^(?:0|[1-9][0-9]?)$/;
const DISPATCH_IDENTIFIER = /^dispatch-[0-9]{4,}$/;

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
  kind: "adk_event";
  author: "user" | "streamline_planner" | "router_planner" | "other";
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
  resume(after: EventCursor): void;
  unsubscribe(): void;
}

export interface OperationsSnapshotCursor {
  generation: string;
  revision: string;
}

export interface OperationsEventSubscription {
  resume(after: OperationsSnapshotCursor): void;
  unsubscribe(): void;
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function exactKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): void {
  const allowed = new Set([...required, ...optional]);
  if (
    required.some((key) => !(key in value)) ||
    Object.keys(value).some((key) => !allowed.has(key))
  ) {
    throw new Error("Event frame has an invalid closed shape");
  }
}

function safeIdentifier(value: unknown, maximum = 256): string {
  if (
    typeof value !== "string" ||
    value.length > maximum ||
    !SAFE_IDENTIFIER.test(value)
  ) {
    throw new Error("Event frame contains an invalid identifier");
  }
  return value;
}

function subscriptionIdentifier(value: unknown): string {
  if (typeof value !== "string" || !SUBSCRIPTION_IDENTIFIER.test(value)) {
    throw new Error("Event frame contains an invalid subscription ID");
  }
  return value;
}

function boundedString(
  value: unknown,
  minimum: number,
  maximum: number,
): string {
  if (
    typeof value !== "string" ||
    value.length < minimum ||
    value.length > maximum
  ) {
    throw new Error("Event frame contains an invalid bounded string");
  }
  return value;
}

function boundedInteger(value: unknown, minimum: number): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new Error("Event frame contains an invalid integer");
  }
  return value as number;
}

function optionalBoolean(value: unknown): boolean | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new Error("Event frame contains an invalid boolean");
  }
  return value;
}

function parseCursor(value: unknown): EventCursor {
  if (!isRecord(value)) {
    throw new Error("Event frame cursor is invalid");
  }
  exactKeys(value, ["generation", "sequence"]);
  const generation = safeIdentifier(value.generation);
  if (
    typeof value.sequence !== "string" ||
    value.sequence.length > 20 ||
    !UNSIGNED_DECIMAL.test(value.sequence)
  ) {
    throw new Error("Event cursor sequence is invalid");
  }
  return { generation, sequence: value.sequence };
}

function parseStream(
  value: unknown,
): { kind: "run"; id: string } | { kind: "operations" } {
  if (!isRecord(value) || typeof value.kind !== "string") {
    throw new Error("Event stream is invalid");
  }
  if (value.kind === "run") {
    exactKeys(value, ["kind", "id"]);
    return { kind: "run", id: safeIdentifier(value.id) };
  }
  if (value.kind === "operations") {
    exactKeys(value, ["kind"]);
    return { kind: "operations" };
  }
  throw new Error("Event stream kind is unknown");
}

function parseSubtask(value: unknown): PlannerPlan["subtasks"][number] {
  if (!isRecord(value)) {
    throw new Error("Planner subtask is invalid");
  }
  exactKeys(value, ["id", "objective", "instructions", "status"]);
  if (typeof value.id !== "string" || !SUBTASK_IDENTIFIER.test(value.id)) {
    throw new Error("Planner subtask ID is invalid");
  }
  const objective = boundedString(value.objective, 1, 8192);
  const instructions = boundedString(value.instructions, 1, 32768);
  if (
    value.status !== "pending" &&
    value.status !== "running" &&
    value.status !== "succeeded" &&
    value.status !== "failed"
  ) {
    throw new Error("Planner subtask status is invalid");
  }
  return { id: value.id, objective, instructions, status: value.status };
}

function parseDispatch(
  value: unknown,
): NonNullable<PlannerPlan["activeDispatch"]> {
  if (!isRecord(value)) {
    throw new Error("Planner dispatch is invalid");
  }
  exactKeys(value, ["callId", "subtaskId", "workerName"]);
  if (
    typeof value.callId !== "string" ||
    value.callId.length > 64 ||
    !DISPATCH_IDENTIFIER.test(value.callId) ||
    typeof value.subtaskId !== "string" ||
    !SUBTASK_IDENTIFIER.test(value.subtaskId)
  ) {
    throw new Error("Planner dispatch identity is invalid");
  }
  return {
    callId: value.callId,
    subtaskId: value.subtaskId,
    workerName: safeIdentifier(value.workerName, 128),
  };
}

function parsePlan(value: unknown): PlannerPlan {
  if (!isRecord(value)) {
    throw new Error("Planner plan is invalid");
  }
  exactKeys(
    value,
    ["revision", "subtasks"],
    ["currentSubtaskId", "activeDispatch"],
  );
  const revision = boundedInteger(value.revision, 1);
  if (
    !Array.isArray(value.subtasks) ||
    value.subtasks.length < 1 ||
    value.subtasks.length > 32
  ) {
    throw new Error("Planner subtask list is invalid");
  }
  const subtasks = value.subtasks.map(parseSubtask);
  if (subtasks.some((subtask, index) => subtask.id !== String(index))) {
    throw new Error("Planner subtask ID order is invalid");
  }
  const identities = new Set(subtasks.map((subtask) => subtask.id));
  let currentSubtaskId: string | undefined;
  if (value.currentSubtaskId !== undefined) {
    if (
      typeof value.currentSubtaskId !== "string" ||
      !SUBTASK_IDENTIFIER.test(value.currentSubtaskId) ||
      !identities.has(value.currentSubtaskId)
    ) {
      throw new Error("Planner current subtask is invalid");
    }
    currentSubtaskId = value.currentSubtaskId;
  }
  const activeDispatch =
    value.activeDispatch === undefined
      ? undefined
      : parseDispatch(value.activeDispatch);
  if (
    activeDispatch !== undefined &&
    !identities.has(activeDispatch.subtaskId)
  ) {
    throw new Error("Planner dispatch selects an unknown subtask");
  }
  const firstUnresolved = subtasks.find(
    (subtask) => subtask.status === "pending" || subtask.status === "running",
  );
  if (currentSubtaskId !== firstUnresolved?.id) {
    throw new Error("Planner current subtask does not match its plan");
  }
  const running = subtasks.filter((subtask) => subtask.status === "running");
  if (activeDispatch === undefined) {
    if (running.length !== 0) {
      throw new Error("Planner running subtask has no active dispatch");
    }
  } else if (
    running.length !== 1 ||
    currentSubtaskId !== activeDispatch.subtaskId ||
    running[0]?.id !== activeDispatch.subtaskId
  ) {
    throw new Error("Planner active dispatch does not match its plan");
  }
  return {
    revision,
    subtasks,
    ...(currentSubtaskId === undefined ? {} : { currentSubtaskId }),
    ...(activeDispatch === undefined ? {} : { activeDispatch }),
  };
}

function parseActivity(value: unknown): PlannerActivity {
  if (!isRecord(value)) {
    throw new Error("Planner activity is invalid");
  }
  exactKeys(
    value,
    ["kind", "author", "functionCalls", "functionResults"],
    [
      "inputTokens",
      "outputTokens",
      "skipSummarization",
      "escalate",
      "truncated",
    ],
  );
  if (value.kind !== "adk_event") {
    throw new Error("Planner activity kind is invalid");
  }
  if (
    value.author !== "user" &&
    value.author !== "streamline_planner" &&
    value.author !== "router_planner" &&
    value.author !== "other"
  ) {
    throw new Error("Planner activity author is invalid");
  }
  function names(source: unknown): string[] {
    if (!Array.isArray(source) || source.length > 32) {
      throw new Error("Planner activity function list is invalid");
    }
    return source.map((name) => safeIdentifier(name));
  }
  const inputTokens =
    value.inputTokens === undefined
      ? undefined
      : boundedInteger(value.inputTokens, 0);
  const outputTokens =
    value.outputTokens === undefined
      ? undefined
      : boundedInteger(value.outputTokens, 0);
  const skipSummarization = optionalBoolean(value.skipSummarization);
  const escalate = optionalBoolean(value.escalate);
  const truncated = optionalBoolean(value.truncated);
  return {
    kind: "adk_event",
    author: value.author,
    functionCalls: names(value.functionCalls),
    functionResults: names(value.functionResults),
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(skipSummarization === undefined ? {} : { skipSummarization }),
    ...(escalate === undefined ? {} : { escalate }),
    ...(truncated === undefined ? {} : { truncated }),
  };
}

function plannerBase(
  value: Record<string, unknown>,
): Omit<PlannerEventData, "eventKind"> {
  return {
    stageExecutionId: safeIdentifier(value.stageExecutionId),
    sessionId: safeIdentifier(value.sessionId),
    invocationId: safeIdentifier(value.invocationId),
  };
}

function parsePlannerEvent(value: unknown): PlannerEventData {
  if (!isRecord(value) || typeof value.eventKind !== "string") {
    throw new Error("Planner event is invalid");
  }
  const common = [
    "stageExecutionId",
    "sessionId",
    "invocationId",
    "eventKind",
  ] as const;
  const eventKind = value.eventKind as PlannerEventKind;
  const base = plannerBase(value);
  switch (eventKind) {
    case "planner.started":
    case "planner.request_recorded":
      exactKeys(value, common);
      return { ...base, eventKind };
    case "planner.activity":
      exactKeys(value, [...common, "activity"]);
      return { ...base, eventKind, activity: parseActivity(value.activity) };
    case "planner.plan_changed": {
      exactKeys(value, [...common, "plan"]);
      const plan = parsePlan(value.plan);
      if (plan.activeDispatch !== undefined) {
        throw new Error("Changed Planner plan cannot contain a dispatch");
      }
      return { ...base, eventKind, plan };
    }
    case "planner.current_changed": {
      exactKeys(value, [...common, "planRevision"], ["subtaskId"]);
      const planRevision = boundedInteger(value.planRevision, 1);
      if (
        value.subtaskId !== undefined &&
        (typeof value.subtaskId !== "string" ||
          !SUBTASK_IDENTIFIER.test(value.subtaskId))
      ) {
        throw new Error("Planner current subtask is invalid");
      }
      return {
        ...base,
        eventKind,
        planRevision,
        ...(value.subtaskId === undefined
          ? {}
          : { subtaskId: value.subtaskId }),
      };
    }
    case "planner.dispatch_selected": {
      exactKeys(value, [
        ...common,
        "planRevision",
        "subtaskId",
        "callId",
        "workerName",
      ]);
      const dispatch = parseDispatch({
        callId: value.callId,
        subtaskId: value.subtaskId,
        workerName: value.workerName,
      });
      return {
        ...base,
        eventKind,
        planRevision: boundedInteger(value.planRevision, 1),
        ...dispatch,
      };
    }
    case "planner.dispatch_started":
    case "planner.dispatch_completed": {
      exactKeys(value, [
        ...common,
        "plan",
        "subtaskId",
        "callId",
        "workerName",
        ...(eventKind === "planner.dispatch_completed" ? ["outcome"] : []),
      ]);
      const plan = parsePlan(value.plan);
      const dispatch = parseDispatch({
        callId: value.callId,
        subtaskId: value.subtaskId,
        workerName: value.workerName,
      });
      if (eventKind === "planner.dispatch_started") {
        if (
          plan.activeDispatch === undefined ||
          JSON.stringify(plan.activeDispatch) !== JSON.stringify(dispatch)
        ) {
          throw new Error("Started dispatch and Planner plan disagree");
        }
        return { ...base, eventKind, plan, ...dispatch };
      }
      if (
        plan.activeDispatch !== undefined ||
        (value.outcome !== "succeeded" && value.outcome !== "failed")
      ) {
        throw new Error("Completed dispatch is invalid");
      }
      return {
        ...base,
        eventKind,
        plan,
        ...dispatch,
        outcome: value.outcome,
      };
    }
    case "planner.finish_requested": {
      exactKeys(value, [...common, "planRevision", "outcome"]);
      if (value.outcome !== "succeeded" && value.outcome !== "failed") {
        throw new Error("Planner finish outcome is invalid");
      }
      const planRevision = boundedInteger(
        value.planRevision,
        value.outcome === "succeeded" ? 1 : 0,
      );
      return { ...base, eventKind, planRevision, outcome: value.outcome };
    }
    case "planner.completed":
      exactKeys(value, [...common, "outcome"]);
      if (value.outcome !== "succeeded" && value.outcome !== "failed") {
        throw new Error("Planner completion outcome is invalid");
      }
      return { ...base, eventKind, outcome: value.outcome };
    case "planner.failed":
      exactKeys(value, [...common, "outcome", "code"]);
      if (value.outcome !== "failure") {
        throw new Error("Planner failure outcome is invalid");
      }
      return {
        ...base,
        eventKind,
        outcome: "failure",
        code: safeIdentifier(value.code),
      };
    default:
      throw new Error("Planner event kind is unknown");
  }
}

function parseLifecycleEvent(value: unknown): LifecycleEventData {
  if (!isRecord(value)) {
    throw new Error("Lifecycle event is invalid");
  }
  exactKeys(value, ["runId", "resource", "state"], ["stageExecutionId"]);
  const runId = safeIdentifier(value.runId);
  const state = safeIdentifier(value.state);
  if (value.resource === "run") {
    if (value.stageExecutionId !== undefined) {
      throw new Error("Run lifecycle event contains a Stage identity");
    }
    return { runId, resource: "run", state };
  }
  if (value.resource !== "stageExecution") {
    throw new Error("Lifecycle resource is unknown");
  }
  return {
    runId,
    resource: "stageExecution",
    stageExecutionId: safeIdentifier(value.stageExecutionId),
    state,
  };
}

function parseOperationsEvent(value: unknown): OperationsEventData {
  if (!isRecord(value)) {
    throw new Error("Operations event is invalid");
  }
  exactKeys(value, ["resource", "revision"], ["resourceId"]);
  if (
    value.resource !== "runtimeAgent" &&
    value.resource !== "allocation" &&
    value.resource !== "configuration" &&
    value.resource !== "credential" &&
    value.resource !== "schedulerSettings"
  ) {
    throw new Error("Operations resource is unknown");
  }
  if (
    typeof value.revision !== "string" ||
    value.revision.length > 20 ||
    !UNSIGNED_DECIMAL.test(value.revision)
  ) {
    throw new Error("Operations revision is invalid");
  }
  const resourceId =
    value.resourceId === undefined
      ? undefined
      : safeIdentifier(value.resourceId);
  if (value.resource === "schedulerSettings" && resourceId !== undefined) {
    throw new Error("Scheduler settings invalidation contains a resource ID");
  }
  return {
    resource: value.resource,
    revision: value.revision,
    ...(resourceId === undefined ? {} : { resourceId }),
  };
}

function parseOccurredAt(value: unknown): string {
  const result = boundedString(value, 1, 64);
  if (Number.isNaN(Date.parse(result))) {
    throw new Error("Event timestamp is invalid");
  }
  return result;
}

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
  if (!isRecord(parsed) || parsed.version !== EVENT_PROTOCOL) {
    throw new Error("Server event protocol version is invalid");
  }
  switch (parsed.type) {
    case "subscribed": {
      exactKeys(parsed, [
        "version",
        "type",
        "subscriptionId",
        "stream",
        "cursor",
      ]);
      return {
        type: "subscribed",
        subscriptionId: subscriptionIdentifier(parsed.subscriptionId),
        stream: parseStream(parsed.stream),
        cursor: parseCursor(parsed.cursor),
      };
    }
    case "unsubscribed":
      exactKeys(parsed, ["version", "type", "subscriptionId"]);
      return {
        type: "unsubscribed",
        subscriptionId: subscriptionIdentifier(parsed.subscriptionId),
      };
    case "event": {
      exactKeys(parsed, [
        "version",
        "type",
        "subscriptionId",
        "stream",
        "cursor",
        "kind",
        "occurredAt",
        "data",
      ]);
      const subscriptionId = subscriptionIdentifier(parsed.subscriptionId);
      const stream = parseStream(parsed.stream);
      const common = {
        type: "event" as const,
        subscriptionId,
        cursor: parseCursor(parsed.cursor),
        occurredAt: parseOccurredAt(parsed.occurredAt),
      };
      if (stream.kind === "operations") {
        if (parsed.kind !== "operations.changed") {
          throw new Error("Operations event kind is unknown");
        }
        return {
          ...common,
          stream,
          kind: "operations.changed",
          data: parseOperationsEvent(parsed.data),
        };
      }
      if (parsed.kind === "planner.event") {
        return {
          ...common,
          stream,
          kind: "planner.event",
          data: parsePlannerEvent(parsed.data),
        };
      }
      if (parsed.kind === "lifecycle.changed") {
        return {
          ...common,
          stream,
          kind: "lifecycle.changed",
          data: parseLifecycleEvent(parsed.data),
        };
      }
      throw new Error("Run event kind is unknown");
    }
    case "resync_required": {
      exactKeys(parsed, [
        "version",
        "type",
        "subscriptionId",
        "stream",
        "reason",
      ]);
      if (
        parsed.reason !== "generation_changed" &&
        parsed.reason !== "cursor_unavailable" &&
        parsed.reason !== "sequence_gap"
      ) {
        throw new Error("Resync reason is unknown");
      }
      return {
        type: "resync_required",
        subscriptionId: subscriptionIdentifier(parsed.subscriptionId),
        stream: parseStream(parsed.stream),
        reason: parsed.reason,
      };
    }
    case "error": {
      exactKeys(
        parsed,
        ["version", "type", "code", "message", "retryable"],
        ["subscriptionId"],
      );
      if (
        parsed.code !== "invalid_frame" &&
        parsed.code !== "subscription_limit" &&
        parsed.code !== "unauthorized" &&
        parsed.code !== "not_found" &&
        parsed.code !== "overloaded"
      ) {
        throw new Error("Event error code is unknown");
      }
      if (typeof parsed.retryable !== "boolean") {
        throw new Error("Event error retryability is invalid");
      }
      return {
        type: "error",
        ...(parsed.subscriptionId === undefined
          ? {}
          : { subscriptionId: subscriptionIdentifier(parsed.subscriptionId) }),
        code: parsed.code,
        message: boundedString(parsed.message, 1, 512),
        retryable: parsed.retryable,
      };
    }
    default:
      throw new Error("Server event frame type is unknown");
  }
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
  parseCursor(cursor);
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
    safeIdentifier(runId);
    validateCursor(after);
    const id = `run-ui-${this.#nextSubscription}`;
    this.#nextSubscription += 1;
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
      resume: (cursor) => {
        if (!active) {
          return;
        }
        validateCursor(cursor);
        record.cursor = copyCursor(cursor);
        delete record.requestedCursor;
        record.phase = "pending";
        callbacks.onStateChange("connecting");
        this.#ensureConnection();
        if (this.#connection?.socket.readyState === 1) {
          this.#sendSubscription(record);
        }
      },
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
    const id = `operations-ui-${this.#nextSubscription}`;
    this.#nextSubscription += 1;
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
      resume: (cursor) => {
        if (!active) {
          return;
        }
        const resumed = {
          generation: cursor.generation,
          sequence: cursor.revision,
        };
        validateCursor(resumed);
        record.cursor = copyCursor(resumed);
        delete record.requestedCursor;
        record.phase = "pending";
        callbacks.onStateChange("connecting");
        this.#ensureConnection();
        if (this.#connection?.socket.readyState === 1) {
          this.#sendSubscription(record);
        }
      },
      unsubscribe: () => {
        if (!active) {
          return;
        }
        active = false;
        this.#unsubscribe(record);
      },
    };
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
      record.phase !== "pending" ||
      record.requestedCursor === undefined ||
      !subscriptionMatchesStream(record, frame.stream) ||
      !sameCursor(frame.cursor, record.requestedCursor)
    ) {
      this.#resyncAll("protocol_error");
      return;
    }
    record.cursor = copyCursor(frame.cursor);
    delete record.requestedCursor;
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
      this.#resyncAll("generation_changed");
      return;
    }
    const current = BigInt(frame.cursor.sequence);
    const previous = BigInt(record.cursor.sequence);
    if (current <= previous) {
      return;
    }
    if (current !== previous + 1n) {
      this.#resyncAll("sequence_gap");
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
          this.#resyncAll("projection_gap");
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
      this.#resyncAll("projection_gap");
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
    this.#resyncAll(frame.reason);
  }

  #errorFrame(frame: ErrorFrame): void {
    if (frame.subscriptionId !== undefined) {
      const record = this.#subscriptions.get(frame.subscriptionId);
      if (record?.phase === "closing" && frame.code === "not_found") {
        this.#subscriptions.delete(record.id);
        this.#closeIfIdle();
        return;
      }
      if (record !== undefined) {
        record.callbacks.onError(frame.message);
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

  #resyncAll(reason: RunResyncReason): void {
    this.#cancelReconnect();
    const connection = this.#connection;
    this.#connection = undefined;
    if (connection !== undefined) {
      connection.socket.close(1002, "authoritative resync required");
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
      record.phase === "subscribed"
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
