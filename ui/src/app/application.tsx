import { QueryClientProvider } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { RouterProvider } from "react-router/dom";
import type { RouterProviderProps } from "react-router/dom";

import { SessionProvider, type SessionAPI } from "../auth/session";
import { createApplicationQueryClient } from "./query-client";

export function Application({
  api,
  router,
}: {
  api: SessionAPI;
  router: RouterProviderProps["router"];
}): ReactNode {
  const [queryClient] = useState(createApplicationQueryClient);
  return (
    <QueryClientProvider client={queryClient}>
      <SessionProvider api={api}>
        <RouterProvider router={router} />
      </SessionProvider>
    </QueryClientProvider>
  );
}
