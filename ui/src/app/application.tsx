import { QueryClientProvider } from "@tanstack/react-query";
import { type ReactNode, useState } from "react";
import { RouterProvider } from "react-router/dom";
import type { RouterProviderProps } from "react-router/dom";

import { SessionProvider, type SessionAPI } from "../auth/session";
import type { PublicAPI } from "../api/client";
import { PublicAPIProvider } from "../api/context";
import { createApplicationQueryClient } from "./query-client";

export function Application({
  api,
  publicAPI,
  router,
}: {
  api: SessionAPI;
  publicAPI?: PublicAPI;
  router: RouterProviderProps["router"];
}): ReactNode {
  const [queryClient] = useState(createApplicationQueryClient);
  return (
    <QueryClientProvider client={queryClient}>
      <PublicAPIProvider
        {...(publicAPI === undefined ? {} : { api: publicAPI })}
      >
        <SessionProvider api={api}>
          <RouterProvider router={router} />
        </SessionProvider>
      </PublicAPIProvider>
    </QueryClientProvider>
  );
}
