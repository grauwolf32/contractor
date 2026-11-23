import { createContext, type ReactNode, useContext } from "react";
import type { SharedConfig } from "../api/shared-config";

const Context = createContext<SharedConfig | null>(null);

export function SharedConfigProvider({
  config,
  children,
}: {
  config: SharedConfig;
  children: ReactNode;
}) {
  return <Context.Provider value={config}>{children}</Context.Provider>;
}

export function useSharedConfig(): SharedConfig {
  const config = useContext(Context);
  if (config === null) throw new Error("Server settings have not been loaded");
  return config;
}
