import type { ReactNode } from "react";

import { RunEventsContext } from "./context";
import type { RunEventsManager } from "./run-events";

export function RunEventsProvider({
  manager,
  children,
}: {
  manager: RunEventsManager;
  children: ReactNode;
}) {
  return (
    <RunEventsContext.Provider value={manager}>
      {children}
    </RunEventsContext.Provider>
  );
}
