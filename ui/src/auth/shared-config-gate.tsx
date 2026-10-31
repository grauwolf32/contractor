import { useQuery } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { usePublicAPI } from "../api/context";
import { startupQueryRetry } from "../app/query-client";
import { getSharedConfig } from "../api/shared-config";
import { SharedConfigProvider } from "../config/shared-config";
import { QueryView } from "../app/query-view";

/** Protected content starts only after the authenticated startup read succeeds. */
export function SharedConfigGate({ children }: { children: ReactNode }) {
  const api = usePublicAPI();
  const query = useQuery({
    queryKey: ["shared-config", api.apiBaseUrl],
    queryFn: ({ signal }) => getSharedConfig(api, signal),
    staleTime: Infinity,
    ...startupQueryRetry,
  });
  const content = (
    <QueryView query={query} errorContext="Could not load Server settings">
      {(config) => (
        <SharedConfigProvider config={config}>{children}</SharedConfigProvider>
      )}
    </QueryView>
  );
  return query.data === undefined ? (
    <main className="ops-state">{content}</main>
  ) : (
    content
  );
}
