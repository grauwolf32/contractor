import { type ReactNode, useEffect, useState } from "react";

import { useSession } from "../auth/session";
import { RunDraftStoreContext } from "./context";
import {
  discardSessionRunDrafts,
  sessionRunDraftStore,
} from "./session-stores";

const ANONYMOUS_OWNER = "anonymous";

/** The session store of `ownerId`'s Run drafts, for one owner at a time. */
function RunDraftOwnerScope({
  ownerId,
  children,
}: {
  ownerId: string;
  children: ReactNode;
}) {
  const [store] = useState(() => sessionRunDraftStore(ownerId));
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
    <RunDraftOwnerScope
      key={userId ?? ANONYMOUS_OWNER}
      ownerId={userId ?? ANONYMOUS_OWNER}
    >
      {children}
    </RunDraftOwnerScope>
  );
}
