import { at, record, textValue, type Draft } from "./document";
import type { StudioGraph, StudioNode } from "./graph";

export const NODE_WIDTH = 218;
export const NODE_HEIGHT = 154;
const COLUMN = 340;
const ROW = 206;
const FOOTERS = new Set(["failure", "execution", "interaction", "routing"]);
const compare = (left: string, right: string) =>
  left < right ? -1 : left > right ? 1 : 0;

/** Arrange dependencies without modifying the authored document. Cycles share
 * a column, so an incomplete draft is still visible and layout always finishes. */
export function layoutGraph(graph: StudioGraph, draft: Draft): StudioGraph {
  const nodes = graph.nodes.filter((node) => !FOOTERS.has(node.type));
  const byID = new Map(nodes.map((node) => [node.id, node]));
  const next = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  const previous = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  for (const edge of graph.edges) {
    if (!byID.has(edge.from) || !byID.has(edge.to)) continue;
    // Workflow progression follows outcomes. Artifact namespaces can be
    // supplied externally or reused; a wire alone does not order two stages.
    if (
      draft.kind === "Workflow" &&
      edge.type === "wire" &&
      byID.get(edge.from)!.type === "stage" &&
      byID.get(edge.to)!.type === "stage"
    )
      continue;
    next.get(edge.from)!.add(edge.to);
    previous.get(edge.to)!.add(edge.from);
  }

  // Tarjan's strongly connected components make the dependency graph acyclic.
  const indices = new Map<string, number>(),
    low = new Map<string, number>(),
    stack: string[] = [],
    active = new Set<string>(),
    components: string[][] = [];
  const visit = (id: string) => {
    const index = indices.size;
    indices.set(id, index);
    low.set(id, index);
    stack.push(id);
    active.add(id);
    for (const target of [...next.get(id)!].sort(compare)) {
      if (!indices.has(target)) {
        visit(target);
        low.set(id, Math.min(low.get(id)!, low.get(target)!));
      } else if (active.has(target))
        low.set(id, Math.min(low.get(id)!, indices.get(target)!));
    }
    if (low.get(id) !== indices.get(id)) return;
    const component: string[] = [];
    let member: string;
    do {
      member = stack.pop()!;
      active.delete(member);
      component.push(member);
    } while (member !== id);
    components.push(component);
  };
  for (const id of [...byID.keys()].sort(compare))
    if (!indices.has(id)) visit(id);
  const componentOf = new Map(
    components.flatMap((members, index) =>
      members.map((id) => [id, index] as const),
    ),
  );
  const minimum = (node: StudioNode): number => {
    if (node.type === "stage" || node.type === "agent") return 1;
    if (node.type === "inventory") return 2;
    if (node.type === "role") {
      const kind = textValue(record(at(draft.value, node.path)).kind);
      return kind === "assessment"
        ? 4
        : kind === "check"
          ? 3
          : kind === "discovery"
            ? 2
            : 1;
    }
    return 0;
  };
  const ranks = new Map<number, number>();
  const rank = (component: number): number => {
    const existing = ranks.get(component);
    if (existing !== undefined) return existing;
    let value = Math.max(
      0,
      ...components[component]!.map((id) => minimum(byID.get(id)!)),
    );
    for (const id of components[component]!)
      for (const source of previous.get(id)!) {
        const dependency = componentOf.get(source)!;
        if (dependency !== component)
          value = Math.max(value, rank(dependency) + 1);
      }
    ranks.set(component, value);
    return value;
  };
  const columns = new Map<number, StudioNode[]>();
  const lastStageColumn = Math.max(
    1,
    ...nodes
      .filter((node) => node.type === "stage")
      .map((node) => rank(componentOf.get(node.id)!)),
  );
  for (const node of nodes) {
    const column =
      node.type === "output"
        ? lastStageColumn + 1
        : rank(componentOf.get(node.id)!);
    columns.set(column, [...(columns.get(column) ?? []), node]);
  }
  const positions = new Map<string, { x: number; y: number }>();
  const entry = textValue(record(draft.value.spec).entryStage);
  const center = (node: StudioNode) => {
    const sources = [...previous.get(node.id)!].flatMap((id) => {
      const position = positions.get(id);
      return position ? [position.y] : [];
    });
    return sources.length
      ? sources.reduce((sum, y) => sum + y, 0) / sources.length
      : 70;
  };
  for (const column of [...columns.keys()].sort((a, b) => a - b)) {
    const members = columns.get(column)!;
    members.sort(
      (a, b) =>
        Number(b.type === "stage" && b.title === entry) -
          Number(a.type === "stage" && a.title === entry) ||
        center(a) - center(b) ||
        compare(a.id, b.id),
    );
    members.forEach((node, row) =>
      positions.set(node.id, { x: 30 + column * COLUMN, y: 70 + row * ROW }),
    );
  }
  const bottom =
    Math.max(70, ...[...positions.values()].map(({ y }) => y)) + ROW;
  graph.nodes
    .filter((node) => FOOTERS.has(node.type))
    .sort((a, b) => compare(a.id, b.id))
    .forEach((node, index) =>
      positions.set(node.id, { x: 30 + (index + 1) * COLUMN, y: bottom }),
    );
  return {
    nodes: graph.nodes.map((node) => ({ ...node, ...positions.get(node.id)! })),
    edges: graph.edges,
  };
}
