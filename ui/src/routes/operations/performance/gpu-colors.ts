import { useState } from "react";

import type { PerformanceHistory } from "../../../api/performance";

// Theme tokens --gpu-1 … --gpu-16 (src/app/theme.css) keep series legible in
// light, dark and black.
const palette = [
  "var(--gpu-1)",
  "var(--gpu-2)",
  "var(--gpu-3)",
  "var(--gpu-4)",
  "var(--gpu-5)",
  "var(--gpu-6)",
  "var(--gpu-7)",
  "var(--gpu-8)",
  "var(--gpu-9)",
  "var(--gpu-10)",
  "var(--gpu-11)",
  "var(--gpu-12)",
  "var(--gpu-13)",
  "var(--gpu-14)",
  "var(--gpu-15)",
  "var(--gpu-16)",
] as const;

export type GPUColors = ReadonlyMap<string, string>;

export function historyGPUDevices(
  history: PerformanceHistory | undefined,
): Map<string, string> {
  const devices = new Map<string, string>();
  // Prefer identities that have observations nearest the end of the range.
  for (const point of [...(history?.points ?? [])].reverse()) {
    for (const device of point.gpu?.devices ?? []) {
      if (devices.size < 8 && !devices.has(device.id))
        devices.set(device.id, device.name);
    }
  }
  return new Map([...devices].sort(([a], [b]) => a.localeCompare(b)));
}

function assignColors(previous: GPUColors, ids: readonly string[]): GPUColors {
  const next = new Map(previous);
  const visibleColors = new Set(ids.map((id) => next.get(id)));
  for (const id of [...new Set(ids)].sort()) {
    if (next.has(id)) continue;
    const color =
      palette.find((candidate) => ![...next.values()].includes(candidate)) ??
      palette.find((candidate) => !visibleColors.has(candidate)) ??
      palette[0];
    // Release only inactive assignments when the palette fills. A returning
    // device then receives a free color instead of colliding with a visible GPU.
    for (const [previousID, previousColor] of next) {
      if (previousColor === color && !ids.includes(previousID))
        next.delete(previousID);
    }
    next.set(id, color);
    visibleColors.add(color);
  }
  // The palette bounds retained assignments to sixteen identities: eight
  // current devices plus eight devices from the selected history range.
  return next;
}

export function useGPUColors(ids: readonly string[]): GPUColors {
  const [colors, setColors] = useState<GPUColors>(() =>
    assignColors(new Map(), ids),
  );
  if (ids.some((id) => !colors.has(id))) {
    const next = assignColors(colors, ids);
    setColors(next);
    return next;
  }
  return colors;
}
