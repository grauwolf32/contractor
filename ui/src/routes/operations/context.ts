import { useOutletContext } from "react-router";

import type { OperationsSnapshot } from "../../api/operations";

export interface OperationsOutletContext {
  snapshot: OperationsSnapshot;
  refresh: () => void;
  refreshing: boolean;
}

export function useOperationsSnapshot(): OperationsOutletContext {
  return useOutletContext<OperationsOutletContext>();
}
