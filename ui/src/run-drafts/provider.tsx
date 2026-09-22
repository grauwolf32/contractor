import { type ReactNode, useEffect, useState } from "react";

import { useSession } from "../auth/session";
import { RunDraftStoreContext } from "./context";
import { RunDraftMemoryStore } from "./memory";
import {
  discardSessionRunDrafts,
  sessionRunDraftStore,
} from "./session-stores";

const ANONYMOUS_OWNER = "anonymous";

export function RunDraftProvider({
  ownerId,
  retain = false,
  children,
}: {
  ownerId: string;
  retain?: boolean;
  children: ReactNode;
}) {
  return (
    <RunDraftOwnerScope key={ownerId} ownerId={ownerId} retain={retain}>
      {children}
    </RunDraftOwnerScope>
  );
}

function RunDraftOwnerScope({
  ownerId,
  retain,
  children,
}: {
  ownerId: string;
  retain: boolean;
  children: ReactNode;
}) {
  const [store] = useState(() =>
    retain ? sessionRunDraftStore(ownerId) : new RunDraftMemoryStore(ownerId),
  );
  return (
    <RunDraftStoreContext.Provider value={store}>
      {children}
    </RunDraftStoreContext.Provider>
  );
}

export function SessionRunDraftProvider({ children }: { children: ReactNode }) {
  const { session } = useSession();
  const userId = session?.principal.userId;
  // Losing the session keeps its drafts; a different user signing in does not.
  useEffect(() => {
    if (userId !== undefined) discardSessionRunDrafts(userId);
  }, [userId]);
  return (
    <RunDraftProvider ownerId={userId ?? ANONYMOUS_OWNER} retain>
      {children}
    </RunDraftProvider>
  );
}
