import type { RuntimeAgentPrincipal } from "../../../api/operations";
import type { StatusTone } from "../../../app/status-tone";
import { shortenId } from "../../../ui";

export const availabilityCopy: Readonly<
  Record<
    RuntimeAgentPrincipal["availability"],
    { label: string; tone: StatusTone }
  >
> = {
  available: { label: "Available", tone: "done" },
  busy: { label: "Busy", tone: "progress" },
  offline: { label: "Offline", tone: "neutral" },
  slot_unavailable: { label: "Slot unavailable", tone: "warning" },
  adapter_capability_mismatch: { label: "Missing adapter", tone: "warning" },
};

/** The short Agent ID the IdChip shows (first 8 and last 4 characters). */
export function shortAgentId(runtimeAgentId: string): string {
  return shortenId(runtimeAgentId);
}

/**
 * Readable card title: the first saved label, else the live process ID, else
 * a generic name with the short Agent ID.
 */
export function agentDisplayName(principal: RuntimeAgentPrincipal): string {
  const label = principal.labels[0];
  if (label !== undefined) return label;
  if (principal.live !== undefined) return principal.live.instanceId;
  return `Agent ${shortAgentId(principal.runtimeAgentId)}`;
}

export function isOfflineIdentity(principal: RuntimeAgentPrincipal): boolean {
  return principal.live === undefined;
}

export function relativeAge(value: string | undefined, now: number): string {
  if (value === undefined) return "Not observed";
  const seconds = Math.max(0, Math.floor((now - Date.parse(value)) / 1000));
  if (!Number.isFinite(seconds)) return "Not observed";
  if (seconds < 5) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function compareByName(
  a: RuntimeAgentPrincipal,
  b: RuntimeAgentPrincipal,
): number {
  return (
    agentDisplayName(a).localeCompare(agentDisplayName(b)) ||
    a.runtimeAgentId.localeCompare(b.runtimeAgentId)
  );
}
