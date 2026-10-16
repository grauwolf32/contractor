import type { StatusTone } from "../app/status-tone";

// The record type makes the compiler keep this list equal to StatusTone.
const tones: Readonly<Record<StatusTone, true>> = {
  done: true,
  partial: true,
  progress: true,
  blocked: true,
  idle: true,
  review: true,
  warning: true,
  success: true,
  info: true,
  neutral: true,
};

/** Every status tone, for tests that cover each one. */
export const STATUS_TONES = Object.keys(tones) as StatusTone[];
