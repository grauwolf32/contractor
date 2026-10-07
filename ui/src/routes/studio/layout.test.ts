import { describe, expect, it } from "vitest";
import { readDraft, type Kind } from "./document";
import { buildGraph, pathID, type StudioGraph } from "./graph";
import { NODE_HEIGHT, NODE_WIDTH } from "./layout";

function graph(kind: Kind, spec: Record<string, unknown>) {
  const draft = readDraft(
    JSON.stringify({ kind, metadata: { name: "sample" }, spec }),
  );
  const source = draft.source;
  const result = buildGraph(draft);
  expect(draft.source).toBe(source);
  for (const node of result.nodes) {
    expect(Number.isFinite(node.x) && Number.isFinite(node.y)).toBe(true);
    expect(node.x).toBeGreaterThanOrEqual(0);
    expect(node.y).toBeGreaterThanOrEqual(0);
    for (const other of result.nodes) {
      if (node.id === other.id) continue;
      expect(
        Math.abs(node.x - other.x) >= NODE_WIDTH ||
          Math.abs(node.y - other.y) >= NODE_HEIGHT,
      ).toBe(true);
    }
  }
  return result;
}
function block(graph: StudioGraph, section: string, name?: string) {
  return graph.nodes.find(
    (node) => node.id === pathID(["spec", section, ...(name ? [name] : [])]),
  )!;
}
function coordinates(graph: StudioGraph) {
  return Object.fromEntries(
    graph.nodes.map((node) => [node.id, [node.x, node.y]]),
  );
}
describe("Studio dependency layout", () => {
  it("orders a branching join by outcomes, independently of YAML mapping order", () => {
    const stages = {
      join: { workflowOutputs: { report: "result" } },
      right: { on: { succeeded: { next: "join" } } },
      left: { on: { succeeded: { next: "join" } } },
      start: {
        on: {
          succeeded: { next: "left" },
          failed: { retry: { maxAttempts: 2, then: { next: "right" } } },
        },
      },
    };
    const spec = {
      inputs: { source: {} },
      parameters: { mode: {} },
      outputs: { report: {} },
      entryStage: "start",
      stages,
    };
    const result = graph("Workflow", spec);
    expect(block(result, "stages", "left").x).toBeGreaterThan(
      block(result, "stages", "start").x,
    );
    expect(block(result, "stages", "left").x).toBe(
      block(result, "stages", "right").x,
    );
    expect(block(result, "stages", "join").x).toBeGreaterThan(
      block(result, "stages", "right").x,
    );
    expect(block(result, "outputs", "report").x).toBeGreaterThan(
      block(result, "stages", "join").x,
    );
    expect(
      coordinates(
        graph("Workflow", {
          ...spec,
          stages: Object.fromEntries(Object.entries(stages).reverse()),
        }),
      ),
    ).toEqual(coordinates(result));
  });
  it("retains cyclic, unreachable and incomplete blocks with dangling edges omitted", () => {
    const result = graph("Workflow", {
      entryStage: "start",
      stages: {
        start: { on: { succeeded: { next: "again" } } },
        again: { on: { succeeded: { next: "start" } } },
        orphan: { on: { succeeded: { next: "missing" } } },
        self: { on: { succeeded: { next: "self" } } },
      },
    });
    expect(result.nodes).toHaveLength(5);
    expect(block(result, "stages", "start").x).toBe(
      block(result, "stages", "again").x,
    );
    expect(result.edges).toHaveLength(3);
  });
  it("keeps progression when a runtime artifact wire points back to an earlier stage", () => {
    const result = graph("Workflow", {
      entryStage: "start",
      stages: {
        start: {
          on: { succeeded: { next: "later" } },
          context: {
            artifacts: { reused: { namespace: "worker", name: "result" } },
          },
        },
        later: {
          result: {
            artifacts: {
              result: { from: { namespace: "worker", name: "result" } },
            },
          },
        },
      },
    });
    expect(result.edges.some((edge) => edge.type === "wire")).toBe(true);
    expect(block(result, "stages", "later").x).toBeGreaterThan(
      block(result, "stages", "start").x,
    );
  });
  it("orders prepare roles, inventory, checks and assessment, including inventory settings", () => {
    const result = graph("AuditProfile", {
      inputs: { source: {} },
      inventory: {
        source: {
          source: "prepare-output",
          role: "prepare",
          name: "inventory",
        },
        settings: {
          source: "prepare-output",
          role: "prepare",
          name: "settings",
        },
        itemWorkflowRole: "check",
      },
      workflows: {
        assess: {
          kind: "assessment",
          inputs: {
            evidence: {
              source: "retained-output",
              role: "discover",
              name: "evidence",
            },
          },
        },
        check: { kind: "check" },
        prepare: {
          kind: "prepare",
          inputs: { source: { source: "audit-input", name: "source" } },
        },
        discover: {
          kind: "discovery",
          inputs: {
            prepared: {
              source: "prepare-output",
              role: "prepare",
              name: "result",
            },
          },
        },
      },
    });
    expect(block(result, "inventory").x).toBeGreaterThan(
      block(result, "workflows", "prepare").x,
    );
    expect(block(result, "workflows", "check").x).toBeGreaterThan(
      block(result, "inventory").x,
    );
    expect(block(result, "workflows", "assess").x).toBeGreaterThan(
      block(result, "workflows", "discover").x,
    );
    expect(
      result.edges.filter((edge) => edge.to === pathID(["spec", "inventory"])),
    ).toHaveLength(2);
  });
  it("packs Agent components without overlaps and retains every toolset and skill", () => {
    const result = graph("AgentTemplate", {
      toolsets: Array.from({ length: 12 }, (_, index) => ({
        ref: `tool-${index}@1`,
      })),
      skills: Array.from({ length: 12 }, (_, index) => ({
        name: `skill-${index}`,
      })),
    });
    expect(result.nodes).toHaveLength(29);
    expect(
      result.nodes.find((node) => node.type === "agent")!.x,
    ).toBeGreaterThan(
      Math.max(
        ...result.nodes
          .filter((node) => node.type !== "agent")
          .map((node) => node.x),
      ),
    );
  });
  it("handles the document limit of 256 stages", () => {
    const result = graph("Workflow", {
      entryStage: "stage-0",
      stages: Object.fromEntries(
        Array.from({ length: 256 }, (_, index) => [
          `stage-${index}`,
          index < 255
            ? { on: { succeeded: { next: `stage-${index + 1}` } } }
            : {},
        ]),
      ),
    });
    expect(result.nodes).toHaveLength(257);
    expect(block(result, "stages", "stage-255").x).toBeGreaterThan(
      block(result, "stages", "stage-0").x,
    );
  });
});
