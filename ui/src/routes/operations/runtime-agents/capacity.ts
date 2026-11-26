import type { RuntimeAgentObservation } from "../../../api/operations";

export function observationsAgree(agent: RuntimeAgentObservation): boolean {
  return (
    agent.reconciliationReason === undefined &&
    agent.pendingReleaseAllocationIds.length === 0 &&
    agent.authoritativeAllocationIds.length ===
      agent.observedAllocations.length &&
    agent.observedAllocations.every((item) =>
      agent.authoritativeAllocationIds.includes(item.allocationId),
    )
  );
}
