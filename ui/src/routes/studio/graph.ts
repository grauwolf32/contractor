import { at, record, textValue, type Draft, type Path } from "./document";
import { continuation, OUTCOMES } from "./validation";
import { layoutGraph } from "./layout";

export interface StudioNode {
  id: string;
  path: Path;
  type: string;
  title: string;
  subtitle: string;
  x: number;
  y: number;
  removable?: boolean;
}
export interface StudioEdge {
  from: string;
  to: string;
  label: string;
  type: "flow" | "wire" | "failure";
}
export const pathID = (path: Path) => JSON.stringify(path);
export interface StudioGraph {
  nodes: StudioNode[];
  edges: StudioEdge[];
}

export function buildGraph(draft: Draft): StudioGraph {
  const nodes: StudioNode[] = [],
    edges: StudioEdge[] = [],
    spec = record(draft.value.spec);
  const node = (
    path: Path,
    type: string,
    title: string,
    subtitle: string,
    removable = false,
  ) => {
    const result = {
      id: pathID(path),
      path,
      type,
      title,
      subtitle,
      x: 0,
      y: 0,
      removable,
    };
    nodes.push(result);
    return result.id;
  };
  const edge = (
    from: Path,
    to: Path,
    label: string,
    type: StudioEdge["type"] = "flow",
  ) => edges.push({ from: pathID(from), to: pathID(to), label, type });
  const files = (section: string) =>
    Object.entries(record(spec[section])).forEach(([name, value]) => {
      const row = record(value);
      node(
        ["spec", section, name],
        section === "parameters"
          ? "parameter"
          : section === "inputs"
            ? "input"
            : "output",
        name,
        section === "parameters"
          ? row.required
            ? "Required parameter"
            : "Optional parameter"
          : Array.isArray(row.mediaTypes)
            ? row.mediaTypes.join(" · ")
            : "File format needed",
        true,
      );
    });
  if (draft.kind === "Workflow") {
    const stages = record(spec.stages),
      names = Object.keys(stages);
    files("inputs");
    files("parameters");
    names.forEach((name) => {
      const stage = record(stages[name]),
        path: Path = ["spec", "stages", name];
      node(
        path,
        "stage",
        name,
        `${name === spec.entryStage ? "Entry · " : ""}${textValue(stage.planner) || "Planner needed"}`,
        true,
      );
      for (const outcome of OUTCOMES) {
        const raw = record(record(stage.on)[outcome]),
          action = continuation(raw);
        const attempt = raw.retry
          ? ` · retry ×${textValue(record(raw.retry).maxAttempts)}`
          : raw.escalate
            ? ` · escalate ×${textValue(record(raw.escalate).maxAttempts)}`
            : "";
        if (action.next !== undefined)
          edge(
            path,
            ["spec", "stages", textValue(action.next)],
            `${outcome}${attempt}`,
            outcome === "succeeded" ? "flow" : "failure",
          );
        else if (action.fail !== undefined)
          edge(path, ["failure"], `${outcome}${attempt}`, "failure");
      }
      for (const [binding, value] of Object.entries(
        record(record(stage.context).artifacts),
      )) {
        const wire = record(value);
        if (wire.namespace === "inputs")
          edge(["spec", "inputs", textValue(wire.name)], path, binding, "wire");
        else
          for (const producer of names) {
            if (producer === name) continue;
            const produces = Object.values(
              record(record(record(stages[producer]).result).artifacts),
            ).some((value) => {
              const from = record(record(value).from);
              return (
                from.namespace === wire.namespace && from.name === wire.name
              );
            });
            if (produces)
              edge(["spec", "stages", producer], path, binding, "wire");
          }
      }
      for (const [output, result] of Object.entries(
        record(stage.workflowOutputs),
      ))
        edge(path, ["spec", "outputs", output], textValue(result), "wire");
    });
    files("outputs");
    node(
      ["failure"],
      "failure",
      "Failure",
      "Shared terminal · retry limits stay on stages",
    );
  } else if (draft.kind === "AuditProfile") {
    files("inputs");
    node(
      ["spec", "inventory"],
      "inventory",
      "Inventory",
      textValue(record(spec.inventory).implementation),
    );
    for (const key of ["source", "settings"]) {
      const source = record(record(spec.inventory)[key]);
      if (source.source === "audit-input")
        edge(
          ["spec", "inputs", textValue(source.name)],
          ["spec", "inventory"],
          `Inventory ${key}`,
          "wire",
        );
      if (source.source === "prepare-output")
        edge(
          ["spec", "workflows", textValue(source.role)],
          ["spec", "inventory"],
          `${textValue(source.name)} → inventory ${key}`,
          "wire",
        );
    }
    Object.entries(record(spec.workflows)).forEach(([name, value]) => {
      const row = record(value),
        path: Path = ["spec", "workflows", name];
      node(
        path,
        "role",
        name,
        `${textValue(row.kind)} · ${textValue(row.ref)}`,
        true,
      );
      if (name === record(spec.inventory).itemWorkflowRole)
        edge(["spec", "inventory"], path, "Items");
      for (const [binding, value] of Object.entries(record(row.inputs))) {
        const wire = record(value);
        if (wire.source === "audit-input")
          edge(["spec", "inputs", textValue(wire.name)], path, binding, "wire");
        if (
          ["retained-output", "prepare-output"].includes(textValue(wire.source))
        )
          edge(
            ["spec", "workflows", textValue(wire.role)],
            path,
            `${textValue(wire.name)} → ${binding}`,
            "wire",
          );
      }
    });
    node(
      ["spec", "execution"],
      "execution",
      "Execution limits",
      `${textValue(record(spec.execution).maxRounds)} rounds · ${textValue(record(spec.execution).batchSize)} per batch`,
    );
    node(
      ["spec", "interaction"],
      "interaction",
      "Review policy",
      textValue(record(spec.interaction).reportAcceptance),
    );
  } else {
    node(
      ["metadata"],
      "agent",
      textValue(record(draft.value.metadata).name),
      textValue(spec.runtime),
    );
    for (const [key, title] of [
      ["modelPolicy", "Model policy"],
      ["instructions", "Instructions"],
      ["sandboxProfile", "Sandbox"],
      ["summarizer", "Summarizer"],
    ] as const) {
      node(
        ["spec", key],
        key,
        title,
        textValue(spec[key]) ||
          textValue(record(spec[key]).ref) ||
          textValue(record(spec[key]).modelPolicy) ||
          "Unconfigured",
      );
      edge(["spec", key], ["metadata"], title, "wire");
    }
    for (const key of ["toolsets", "skills"]) {
      const values = spec[key];
      if (!Array.isArray(values)) continue;
      values.forEach((value, index) => {
        node(
          ["spec", key, index],
          key === "toolsets" ? "toolset" : "skill",
          textValue(record(value).ref) ||
            textValue(record(value).name) ||
            `${key} ${index + 1}`,
          key === "toolsets" ? "Selected tools" : "Skill file",
          true,
        );
        edge(
          ["spec", key, index],
          ["metadata"],
          key === "toolsets" ? "Tools" : "Skill",
          "wire",
        );
      });
    }
  }
  return layoutGraph(
    {
      nodes,
      edges: edges.filter(
        (edge) =>
          nodes.some((node) => node.id === edge.from) &&
          nodes.some((node) => node.id === edge.to),
      ),
    },
    draft,
  );
}

export function nodeValue(draft: Draft, node: StudioNode): unknown {
  return at(draft.value, node.path);
}
