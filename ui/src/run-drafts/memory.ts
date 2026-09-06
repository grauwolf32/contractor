import type { ArtifactMetadata } from "../api/artifacts";
import type { RunMetadataLabelDraft } from "../api/run-metadata-labels";
import { RunDraftKeyring } from "./idempotency";
import {
  emptyExecutionOverrides,
  type ExecutionOverrideDraft,
} from "./validation";

export const MAX_RETAINED_RUN_DRAFTS = 16;

export interface RunDraftIdentity {
  workflowName: string;
  workflowVersion: string;
  projectId?: string;
}

export interface RunDraftState {
  parameters: Record<string, string | undefined>;
  runtimeLabels: string[];
  metadataLabels: RunMetadataLabelDraft[];
  metadataLabelInput: string;
  artifactSelections: Record<string, string>;
  knownArtifacts: ArtifactMetadata[];
  overrides: ExecutionOverrideDraft;
}

export interface RunDraftSummary extends RunDraftIdentity {
  key: string;
  updatedAt: string;
  ambiguousSubmission: boolean;
}

export interface RunDraftEntry {
  readonly key: string;
  readonly generation: number;
  readonly identity: RunDraftIdentity;
  readonly keyring: RunDraftKeyring;
  state: RunDraftState;
  ambiguousSubmission: boolean;
  activeConsumers: number;
  current: boolean;
  meaningful: boolean;
  updatedAt: string;
}

export type RunDraftAcquisition =
  | { kind: "acquired"; entry: RunDraftEntry }
  | { kind: "capacity"; drafts: RunDraftSummary[] };

function identityKey(identity: RunDraftIdentity): string {
  return JSON.stringify([
    identity.workflowName,
    identity.workflowVersion,
    identity.projectId ?? null,
  ]);
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, child]) => child !== undefined)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, canonicalValue(child)]),
  );
}

function sameState(left: RunDraftState, right: RunDraftState): boolean {
  return (
    JSON.stringify(canonicalValue(left)) ===
    JSON.stringify(canonicalValue(right))
  );
}

export function initialRunDraftState(
  artifactSelections: Readonly<Record<string, string>> = {},
): RunDraftState {
  return {
    parameters: {},
    runtimeLabels: [],
    metadataLabels: [],
    metadataLabelInput: "",
    artifactSelections: { ...artifactSelections },
    knownArtifacts: [],
    overrides: emptyExecutionOverrides(),
  };
}

function cloneState(state: RunDraftState): RunDraftState {
  return structuredClone(state);
}

interface InternalRunDraftEntry extends RunDraftEntry {
  initialState: RunDraftState;
}

export class RunDraftMemoryStore {
  readonly ownerId: string;
  readonly #entries = new Map<string, InternalRunDraftEntry>();
  #nextGeneration = 0;

  constructor(ownerId: string) {
    this.ownerId = ownerId;
  }

  acquire(
    identity: RunDraftIdentity,
    initialState: RunDraftState,
  ): RunDraftAcquisition {
    const key = identityKey(identity);
    const existing = this.#entries.get(key);
    if (existing !== undefined) {
      return { kind: "acquired", entry: existing };
    }
    const retainedCount = Array.from(this.#entries.values()).filter(
      (entry) => entry.meaningful,
    ).length;
    if (retainedCount >= MAX_RETAINED_RUN_DRAFTS) {
      return { kind: "capacity", drafts: this.summaries() };
    }
    const now = new Date().toISOString();
    this.#nextGeneration += 1;
    const entry: InternalRunDraftEntry = {
      key,
      generation: this.#nextGeneration,
      identity: { ...identity },
      keyring: new RunDraftKeyring(),
      state: cloneState(initialState),
      initialState: cloneState(initialState),
      ambiguousSubmission: false,
      activeConsumers: 0,
      current: true,
      meaningful: false,
      updatedAt: now,
    };
    this.#entries.set(key, entry);
    return { kind: "acquired", entry };
  }

  retain(entry: RunDraftEntry): void {
    if (!this.isCurrent(entry)) return;
    entry.activeConsumers += 1;
  }

  release(entry: RunDraftEntry): void {
    if (!this.isCurrent(entry)) return;
    entry.activeConsumers = Math.max(0, entry.activeConsumers - 1);
    queueMicrotask(() => {
      if (
        this.isCurrent(entry) &&
        entry.activeConsumers === 0 &&
        !entry.meaningful
      ) {
        this.#entries.delete(entry.key);
        entry.current = false;
      }
    });
  }

  replaceState(entry: RunDraftEntry, state: RunDraftState): boolean {
    const internal = this.#entries.get(entry.key);
    if (internal !== entry || !entry.current) return false;
    entry.state = cloneState(state);
    entry.meaningful =
      entry.ambiguousSubmission ||
      entry.keyring.hasSubmission() ||
      !sameState(entry.state, internal.initialState);
    entry.updatedAt = new Date().toISOString();
    return true;
  }

  markSubmitted(entry: RunDraftEntry): boolean {
    if (!this.isCurrent(entry)) return false;
    entry.meaningful = true;
    entry.updatedAt = new Date().toISOString();
    return true;
  }

  setAmbiguousSubmission(entry: RunDraftEntry, ambiguous: boolean): boolean {
    const internal = this.#entries.get(entry.key);
    if (internal !== entry || !entry.current) return false;
    entry.ambiguousSubmission = ambiguous;
    entry.meaningful =
      ambiguous ||
      entry.keyring.hasSubmission() ||
      !sameState(entry.state, internal.initialState);
    entry.updatedAt = new Date().toISOString();
    return true;
  }

  discard(entryOrKey: RunDraftEntry | string): void {
    const key = typeof entryOrKey === "string" ? entryOrKey : entryOrKey.key;
    const entry = this.#entries.get(key);
    if (entry === undefined) return;
    this.#entries.delete(key);
    entry.current = false;
  }

  isCurrent(entry: RunDraftEntry): boolean {
    return this.#entries.get(entry.key) === entry && entry.current;
  }

  summaries(): RunDraftSummary[] {
    return Array.from(this.#entries.values())
      .filter((entry) => entry.meaningful)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
      .map((entry) => ({
        key: entry.key,
        ...entry.identity,
        updatedAt: entry.updatedAt,
        ambiguousSubmission: entry.ambiguousSubmission,
      }));
  }
}
