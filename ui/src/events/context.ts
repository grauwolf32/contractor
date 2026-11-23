import { createContext, useContext } from "react";

import type { RunEventsManager } from "./run-events";

export const RunEventsContext = createContext<RunEventsManager | null>(null);

export function useRunEvents(): RunEventsManager {
  const manager = useContext(RunEventsContext);
  if (manager === null) {
    throw new Error("Run event transport is not configured");
  }
  return manager;
}
