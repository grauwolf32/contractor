import type {
  OperationsSnapshotCursor,
  RunResyncReason,
} from "../../events/run-events";

interface LiveRefreshOptions {
  initial: OperationsSnapshotCursor;
  refreshSnapshot: () => Promise<OperationsSnapshotCursor | undefined>;
  refreshPrincipals: () => Promise<unknown>;
  resume: (cursor: OperationsSnapshotCursor) => void;
  random?: () => number;
}

const REFRESH_INTERVAL_MS = 1_000;

/** Serializes REST reads without discarding the newest event revision. */
export class OperationsLiveRefresh {
  readonly #generation: string;
  readonly #refreshSnapshot: LiveRefreshOptions["refreshSnapshot"];
  readonly #refreshPrincipals: LiveRefreshOptions["refreshPrincipals"];
  readonly #resume: LiveRefreshOptions["resume"];
  readonly #random: () => number;
  #appliedRevision: bigint;
  #observedRevision: bigint;
  #snapshotTimer: ReturnType<typeof setTimeout> | undefined;
  #principalTimer: ReturnType<typeof setTimeout> | undefined;
  #snapshotRunning = false;
  #principalRunning = false;
  #principalDirty = false;
  #resyncPending = false;
  #resyncDelay = 0;
  #generationResyncs = 0;
  #disposed = false;

  constructor(options: LiveRefreshOptions) {
    this.#generation = options.initial.generation;
    this.#appliedRevision = BigInt(options.initial.revision);
    this.#observedRevision = this.#appliedRevision;
    this.#refreshSnapshot = options.refreshSnapshot;
    this.#refreshPrincipals = options.refreshPrincipals;
    this.#resume = options.resume;
    this.#random = options.random ?? Math.random;
  }

  event(cursor: OperationsSnapshotCursor, resource: string): void {
    if (this.#disposed || cursor.generation !== this.#generation) return;
    const revision = BigInt(cursor.revision);
    if (revision > this.#observedRevision) this.#observedRevision = revision;
    this.#generationResyncs = 0;
    if (this.#observedRevision > this.#appliedRevision) {
      this.#scheduleSnapshot(REFRESH_INTERVAL_MS);
    }
    if (resource === "runtimeAgent") this.principalsChanged();
  }

  snapshot(cursor: OperationsSnapshotCursor): void {
    if (this.#disposed || cursor.generation !== this.#generation) return;
    const revision = BigInt(cursor.revision);
    if (revision > this.#appliedRevision) this.#appliedRevision = revision;
    if (this.#observedRevision > this.#appliedRevision) {
      this.#scheduleSnapshot(REFRESH_INTERVAL_MS);
    }
  }

  principalsChanged(): void {
    if (this.#disposed) return;
    this.#principalDirty = true;
    if (this.#principalTimer === undefined && !this.#principalRunning) {
      this.#principalTimer = setTimeout(
        () => this.#runPrincipals(),
        REFRESH_INTERVAL_MS,
      );
    }
  }

  resync(reason: RunResyncReason): void {
    if (this.#disposed) return;
    this.#resyncPending = true;
    // A resync needs a new baseline, not a replay of revisions from the old
    // stream. Any fetch already in flight may have started before the resync.
    this.#observedRevision = this.#appliedRevision;
    if (reason === "generation_changed") {
      const maximum = Math.min(30_000, 500 * 2 ** this.#generationResyncs);
      this.#generationResyncs += 1;
      this.#resyncDelay = Math.floor(
        maximum * (0.5 + Math.max(0, Math.min(1, this.#random())) / 2),
      );
    } else {
      this.#resyncDelay = 0;
    }
    if (this.#snapshotTimer !== undefined) {
      clearTimeout(this.#snapshotTimer);
      this.#snapshotTimer = undefined;
    }
    this.#scheduleSnapshot(this.#resyncDelay);
    this.principalsChanged();
  }

  dispose(): void {
    this.#disposed = true;
    if (this.#snapshotTimer !== undefined) clearTimeout(this.#snapshotTimer);
    if (this.#principalTimer !== undefined) clearTimeout(this.#principalTimer);
    this.#snapshotTimer = undefined;
    this.#principalTimer = undefined;
  }

  #scheduleSnapshot(delay: number): void {
    if (
      this.#disposed ||
      this.#snapshotTimer !== undefined ||
      this.#snapshotRunning
    )
      return;
    this.#snapshotTimer = setTimeout(() => {
      this.#snapshotTimer = undefined;
      void this.#runSnapshot();
    }, delay);
  }

  async #runSnapshot(): Promise<void> {
    if (this.#disposed) return;
    this.#snapshotRunning = true;
    const resumeAfterFetch = this.#resyncPending;
    this.#resyncPending = false;
    try {
      const cursor = await this.#refreshSnapshot();
      if (this.#disposed) return;
      if (cursor !== undefined) {
        this.snapshot(cursor);
        if (resumeAfterFetch && cursor.generation === this.#generation) {
          this.#resume(cursor);
        }
      } else if (resumeAfterFetch) {
        this.#resyncPending = true;
      }
    } catch {
      if (resumeAfterFetch) this.#resyncPending = true;
    } finally {
      this.#snapshotRunning = false;
      if (this.#resyncPending) this.#scheduleSnapshot(this.#resyncDelay);
      else if (this.#observedRevision > this.#appliedRevision)
        this.#scheduleSnapshot(REFRESH_INTERVAL_MS);
    }
  }

  async #runPrincipals(): Promise<void> {
    this.#principalTimer = undefined;
    if (this.#disposed || !this.#principalDirty) return;
    this.#principalRunning = true;
    this.#principalDirty = false;
    try {
      await this.#refreshPrincipals();
    } catch {
      // The owning query exposes the error; later events still request a read.
    } finally {
      this.#principalRunning = false;
      if (this.#principalDirty && !this.#disposed) this.principalsChanged();
    }
  }
}
