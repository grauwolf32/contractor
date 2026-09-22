import { RunDraftMemoryStore } from "./memory";

// Session drafts outlive an expired session, so signing in again as the same
// user restores them together with ambiguous-submission idempotency keys.
const sessionStores = new Map<string, RunDraftMemoryStore>();

export function sessionRunDraftStore(ownerId: string): RunDraftMemoryStore {
  let store = sessionStores.get(ownerId);
  if (store === undefined) {
    store = new RunDraftMemoryStore(ownerId);
    sessionStores.set(ownerId, store);
  }
  return store;
}

/** Drops retained session drafts of every owner except `keep`. */
export function discardSessionRunDrafts(keep?: string): void {
  for (const ownerId of [...sessionStores.keys()]) {
    if (ownerId !== keep) sessionStores.delete(ownerId);
  }
}
