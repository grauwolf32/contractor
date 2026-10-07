import type { StatusTone } from "../../app/status-tone";

/**
 * Tones of the Server's operational states: Runtime slots and processes,
 * allocation phases, Stage outcomes, measurement and freshness states. The
 * chip keeps the Server's own word, so an unknown state still reads true.
 */
const STATE_TONES: Readonly<Record<string, StatusTone>> = {
  idle: "idle",
  pending: "idle",
  allocated: "progress",
  busy: "progress",
  reserved: "progress",
  preparing: "progress",
  active: "progress",
  prepared: "progress",
  finalizing: "progress",
  releasing: "progress",
  running: "progress",
  draining: "warning",
  aborting: "warning",
  stale: "warning",
  interrupted: "warning",
  partial: "partial",
  fenced: "blocked",
  failed: "blocked",
  unavailable: "blocked",
  ok: "done",
  available: "done",
  complete: "done",
  succeeded: "done",
  absent: "neutral",
  cancelled: "neutral",
  disabled: "neutral",
  unsupported: "neutral",
};

/** The status tone of an operational state word; unknown words are neutral. */
export function operationsTone(state: string): StatusTone {
  return Object.hasOwn(STATE_TONES, state) ? STATE_TONES[state]! : "neutral";
}
