import { QueryClientProvider } from "@tanstack/react-query";
import { type ReactNode, useEffect, useState } from "react";
import { RouterProvider } from "react-router/dom";
import type { RouterProviderProps } from "react-router/dom";

import { SessionProvider, type SessionAPI } from "../auth/session";
import type { PublicAPI } from "../api/client";
import { PublicAPIProvider } from "../api/context";
import { RunEventsProvider } from "../events/provider";
import { RunEventsManager } from "../events/run-events";
import { SessionRunDraftProvider } from "../run-drafts/provider";
import { createApplicationQueryClient } from "./query-client";

export function Application({
  api,
  publicAPI,
  runEvents,
  router,
}: {
  api: SessionAPI;
  publicAPI: PublicAPI;
  runEvents?: RunEventsManager;
  router: RouterProviderProps["router"];
}): ReactNode {
  const [queryClient] = useState(createApplicationQueryClient);
  const [eventManager] = useState(
    () => runEvents ?? new RunEventsManager(publicAPI.apiBaseUrl),
  );
  useEffect(
    () => () => {
      eventManager.close();
    },
    [eventManager],
  );
  return (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider api={publicAPI}>
        <RunEventsProvider manager={eventManager}>
          <SessionProvider api={api} publicAPI={publicAPI}>
            <SessionRunDraftProvider>
              <RouterProvider router={router} />
            </SessionRunDraftProvider>
          </SessionProvider>
        </RunEventsProvider>
      </PublicAPIProvider>
    </QueryClientProvider>
  );
}
