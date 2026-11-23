import { createContext, type ReactNode, useContext } from "react";

import type { PublicAPI } from "./client";

const PublicAPIContext = createContext<PublicAPI | null>(null);

export function PublicAPIProvider({
  api,
  children,
}: {
  api?: PublicAPI;
  children: ReactNode;
}) {
  return (
    <PublicAPIContext.Provider value={api ?? null}>
      {children}
    </PublicAPIContext.Provider>
  );
}

export function usePublicAPI(): PublicAPI {
  const api = useContext(PublicAPIContext);
  if (api === null) {
    throw new Error("Public API transport is not configured");
  }
  return api;
}
