import { createContext, useContext } from "react";

import { RunDraftMemoryStore } from "./memory";

export const RunDraftStoreContext = createContext<RunDraftMemoryStore | null>(
  null,
);

export function useRunDraftStore(): RunDraftMemoryStore {
  const store = useContext(RunDraftStoreContext);
  if (store === null) {
    throw new Error("useRunDraftStore must be used inside RunDraftProvider");
  }
  return store;
}
