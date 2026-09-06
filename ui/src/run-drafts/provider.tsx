import { type ReactNode, useState } from "react";

import { useSession } from "../auth/session";
import { RunDraftStoreContext } from "./context";
import { RunDraftMemoryStore } from "./memory";

export function RunDraftProvider({
  ownerId,
  children,
}: {
  ownerId: string;
  children: ReactNode;
}) {
  return (
    <RunDraftOwnerScope key={ownerId} ownerId={ownerId}>
      {children}
    </RunDraftOwnerScope>
  );
}

function RunDraftOwnerScope({
  ownerId,
  children,
}: {
  ownerId: string;
  children: ReactNode;
}) {
  const [store] = useState(() => new RunDraftMemoryStore(ownerId));
  return (
    <RunDraftStoreContext.Provider value={store}>
      {children}
    </RunDraftStoreContext.Provider>
  );
}

export function SessionRunDraftProvider({ children }: { children: ReactNode }) {
  const { session } = useSession();
  const ownerId = session?.principal.userId ?? "anonymous";
  return <RunDraftProvider ownerId={ownerId}>{children}</RunDraftProvider>;
}
