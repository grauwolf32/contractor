import type { RunSummary } from "../../api/runs";

export interface EvaluationRunGroup {
  id: string;
  names: string[];
  runs: RunSummary[];
}

export function groupEvaluationRuns(
  runs: readonly RunSummary[],
): EvaluationRunGroup[] {
  const grouped = new Map<string, { names: Set<string>; runs: RunSummary[] }>();
  for (const run of runs) {
    const id = run.labels["eval.id"] ?? "";
    const current = grouped.get(id) ?? { names: new Set(), runs: [] };
    const name = run.labels["eval.name"];
    if (name !== undefined) {
      current.names.add(name);
    }
    current.runs.push(run);
    grouped.set(id, current);
  }
  return [...grouped.entries()]
    .sort(([left], [right]) => {
      if (left === "") return 1;
      if (right === "") return -1;
      return left.localeCompare(right);
    })
    .map(([id, value]) => ({
      id,
      names: [...value.names].sort(),
      runs: value.runs,
    }));
}
