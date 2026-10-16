import { useState } from "react";
import type { WorkflowSummary } from "../../api/workflows";
import { compareVersions } from "../../app/format";

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
        compareVersions(b.ref.version, a.ref.version),
      ),
    }));
}

/**
 * Whether every version of a family is a dotted number, so "Latest" and
 * "Earlier version" follow publication order (compareVersions).
 */
export function hasNumericVersions(
  versions: readonly WorkflowSummary[],
): boolean {
  return versions.every((item) => /^\d+(?:\.\d+)*$/.test(item.ref.version));
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
