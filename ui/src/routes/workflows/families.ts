import { useState } from "react";
import type { WorkflowSummary } from "../../api/workflows";

// Published numeric versions have an explicit UI order, independent of page order.
// Other version labels remain selectable but do not imply publication chronology.
export function compareWorkflowVersions(a: string, b: string): number {
  const numeric = /^\d+(?:\.\d+)*$/;
  if (numeric.test(a) && numeric.test(b)) {
    const left = a.split(".").map(BigInt);
    const right = b.split(".").map(BigInt);
    for (let i = 0; i < Math.max(left.length, right.length); i++) {
      const x = left[i] ?? 0n;
      const y = right[i] ?? 0n;
      if (x !== y) return x > y ? 1 : -1;
    }
  }
  return (
    a.localeCompare(b, "en", { numeric: true }) || (a < b ? -1 : a > b ? 1 : 0)
  );
}

export function groupWorkflowVersions(items: readonly WorkflowSummary[]) {
  const groups = new Map<string, Map<string, WorkflowSummary>>();
  for (const item of items) {
    const versions = groups.get(item.ref.name) ?? new Map();
    versions.set(item.ref.version, item);
    groups.set(item.ref.name, versions);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, versions]) => ({
      name,
      versions: [...versions.values()].sort((a, b) =>
        compareWorkflowVersions(b.ref.version, a.ref.version),
      ),
    }));
}

export function useWorkflowFamilies(items: readonly WorkflowSummary[]) {
  const [choices, setChoices] = useState<Record<string, string>>({});
  return {
    families: groupWorkflowVersions(items).map((family) => ({
      ...family,
      workflow:
        family.versions.find((w) => w.ref.version === choices[family.name]) ??
        family.versions[0]!,
    })),
    selectVersion: (name: string, version: string) =>
      setChoices((old) => ({ ...old, [name]: version })),
  };
}
