import { at, record, textValue, type Draft, type Path } from "./document";
import { validateAdvanced } from "./advanced-validation";

export interface Problem {
  severity: "error" | "warning";
  path: Path;
  message: string;
}
export const OUTCOMES = ["succeeded", "failed", "interrupted"] as const;
export function continuation(value: unknown): Record<string, unknown> {
  const action = record(value);
  return Object.hasOwn(action, "retry")
    ? record(record(action.retry).then)
    : Object.hasOwn(action, "escalate")
      ? record(record(action.escalate).then)
      : action;
}
export function nextStages(stage: unknown): string[] {
  return OUTCOMES.flatMap((outcome) => {
    const next = continuation(record(record(stage).on)[outcome]).next;
    return typeof next === "string" ? [next] : [];
  });
}

/** Rules available from authored YAML. Bundle/catalog resolution stays in CLI. */
export function validateDraft(draft: Draft): Problem[] {
  const problems: Problem[] = [];
  const error = (path: Path, message: string) =>
    problems.push({ severity: "error", path, message });
  const warning = (path: Path, message: string) =>
    problems.push({ severity: "warning", path, message });
  const requiredString = (path: Path) => {
    if (
      typeof at(draft.value, path) !== "string" ||
      !textValue(at(draft.value, path)).trim()
    )
      error(path, "A non-empty string is required.");
  };
  const selector = (path: Path) => {
    if (
      !/^[A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.+-]*$/.test(
        textValue(at(draft.value, path)),
      )
    )
      error(path, "Use a versioned selector such as worker@1.");
  };
  if (draft.value.apiVersion !== "contractor/v1alpha1")
    error(["apiVersion"], "Expected contractor/v1alpha1.");
  for (const key of ["name", "version"]) {
    const value = at(draft.value, ["metadata", key]);
    if (
      typeof value !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9_.+-]*$/.test(value)
    )
      error(["metadata", key], "Use a non-empty identifier string.");
  }
  if (
    draft.value.spec === null ||
    typeof draft.value.spec !== "object" ||
    Array.isArray(draft.value.spec)
  )
    error(["spec"], "A spec mapping is required.");
  const spec = record(draft.value.spec);
  const slots = (path: Path, allowEmpty = true) => {
    const value = at(draft.value, path);
    if (value === undefined && allowEmpty) return;
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      error(path, "Expected a mapping.");
      return;
    }
    if (!allowEmpty && !Object.keys(record(value)).length)
      error(path, "At least one entry is required.");
    for (const [name, slot] of Object.entries(record(value))) {
      const p = [...path, name],
        row = record(slot);
      if (typeof row.required !== "boolean")
        error([...p, "required"], "Choose whether this file is required.");
      if (
        !Array.isArray(row.mediaTypes) ||
        !row.mediaTypes.length ||
        row.mediaTypes.some(
          (type) => typeof type !== "string" || !/^\S+\/\S+$/.test(type),
        )
      )
        error([...p, "mediaTypes"], "Provide one or more media types.");
    }
  };
  if (draft.kind === "Workflow") {
    slots(["spec", "inputs"]);
    slots(["spec", "outputs"]);
    const stages = record(spec.stages),
      names = Object.keys(stages),
      entry = textValue(spec.entryStage);
    if (!names.length) error(["spec", "stages"], "Add at least one stage.");
    if (!Object.hasOwn(stages, entry))
      error(["spec", "entryStage"], "Entry stage must name an existing stage.");
    const adjacency = new Map<string, string[]>();
    for (const [name, value] of Object.entries(stages)) {
      const stage = record(value),
        p: Path = ["spec", "stages", name];
      requiredString([...p, "objective"]);
      selector([...p, "planner"]);
      if (
        stage.session !== undefined &&
        !["isolated", "shared"].includes(textValue(stage.session))
      )
        error([...p, "session"], "Session must be isolated or shared.");
      const agents = record(stage.agents),
        count = Object.keys(agents).length;
      if (
        !count ||
        (["passthrough@1", "streamline@1"].includes(textValue(stage.planner)) &&
          count !== 1)
      )
        error(
          [...p, "agents"],
          "This planner requires exactly one agent; other planners need at least one.",
        );
      for (const agent of Object.keys(agents))
        selector([...p, "agents", agent, "template"]);
      slots([...p, "result", "artifacts"]);
      const exports = record(record(record(stage.context).workspace).export);
      const namespaces = Object.entries(agents).map(
        ([name, agent]) => textValue(record(agent).namespace) || name,
      );
      for (const [resultName, value] of Object.entries(
        record(record(stage.result).artifacts),
      )) {
        const from = record(record(value).from);
        if ([exports.state, exports.diff].includes(resultName)) {
          if (record(value).from !== undefined)
            error(
              [...p, "result", "artifacts", resultName, "from"],
              "Runtime workspace exports must omit from.",
            );
        } else if (
          stage.planner !== "scan-plan@1" &&
          (!textValue(from.name) ||
            !namespaces.includes(textValue(from.namespace)))
        ) {
          error(
            [...p, "result", "artifacts", resultName, "from"],
            "Select a file in a namespace assigned to a stage agent.",
          );
        }
      }
      for (const [binding, value] of Object.entries(
        record(record(stage.context).artifacts),
      )) {
        const row = record(value);
        if (
          !textValue(row.namespace) ||
          !textValue(row.name) ||
          typeof row.required !== "boolean"
        )
          error(
            [...p, "context", "artifacts", binding],
            "A wire needs namespace, name and required.",
          );
        if (
          row.namespace === "inputs" &&
          !Object.hasOwn(record(spec.inputs), textValue(row.name))
        )
          error(
            [...p, "context", "artifacts", binding],
            "This workflow input is not declared.",
          );
      }
      const transitions = record(stage.on);
      const checkAction = (
        raw: unknown,
        path: Path,
        allowed: string[],
        nested = false,
      ) => {
        const action = record(raw),
          keys = Object.keys(action);
        if (keys.length !== 1 || !allowed.includes(keys[0]!)) {
          error(path, `Choose one of ${allowed.join(", ")}.`);
          return;
        }
        if (
          Object.hasOwn(action, "next") &&
          !Object.hasOwn(stages, textValue(action.next))
        )
          error([...path, "next"], "Transition target does not exist.");
        for (const type of ["retry", "escalate"]) {
          if (nested || !Object.hasOwn(action, type)) continue;
          const branch = record(action[type]);
          if (
            !Number.isSafeInteger(branch.maxAttempts) ||
            Number(branch.maxAttempts) < (type === "retry" ? 2 : 1)
          )
            error(
              [...path, type, "maxAttempts"],
              `Must be at least ${type === "retry" ? 2 : 1}.`,
            );
          const then = record(branch.then),
            thenKeys = Object.keys(then);
          if (thenKeys.length !== 1 || !["next", "fail"].includes(thenKeys[0]!))
            error(
              [...path, type, "then"],
              "Continuation must select next or fail.",
            );
          else if (
            then.next !== undefined &&
            !Object.hasOwn(stages, textValue(then.next))
          )
            error(
              [...path, type, "then", "next"],
              "Transition target does not exist.",
            );
          if (
            type === "escalate" &&
            !Object.keys(record(branch.executionConfig)).length
          )
            error(
              [...path, type, "executionConfig"],
              "An escalation execution configuration is required.",
            );
        }
      };
      for (const outcome of OUTCOMES)
        checkAction(
          transitions[outcome],
          [...p, "on", outcome],
          outcome === "succeeded"
            ? ["next", "succeed"]
            : ["next", "retry", "escalate", "fail"],
        );
      adjacency.set(
        name,
        nextStages(stage).filter((target) => Object.hasOwn(stages, target)),
      );
      for (const [output, result] of Object.entries(
        record(stage.workflowOutputs),
      )) {
        if (!Object.hasOwn(record(spec.outputs), output))
          error(
            [...p, "workflowOutputs", output],
            "This workflow output is not declared.",
          );
        const resultSlot = record(
          record(record(stage.result).artifacts)[textValue(result)],
        );
        if (!Object.keys(resultSlot).length)
          error(
            [...p, "workflowOutputs", output],
            "This stage result is not declared.",
          );
        else if (
          !Array.isArray(resultSlot.mediaTypes) ||
          !resultSlot.mediaTypes.some((type) => {
            const types = record(record(spec.outputs)[output]).mediaTypes;
            return Array.isArray(types) && types.includes(type);
          })
        )
          error(
            [...p, "workflowOutputs", output],
            "Result and output media types must intersect.",
          );
      }
    }
    const reached = new Set<string>(),
      visiting = new Set<string>(),
      visited = new Set<string>(),
      order: string[] = [];
    let cyclic = false;
    const walk = (name: string) => {
      if (visiting.has(name)) {
        cyclic = true;
        error(["spec", "stages", name, "on"], "Next transitions form a cycle.");
        return;
      }
      if (visited.has(name)) return;
      visiting.add(name);
      for (const target of adjacency.get(name) ?? []) walk(target);
      visiting.delete(name);
      visited.add(name);
      order.unshift(name);
    };
    names.forEach(walk);
    const reach = (name: string) => {
      if (reached.has(name)) return;
      reached.add(name);
      (adjacency.get(name) ?? []).forEach(reach);
    };
    if (Object.hasOwn(stages, entry)) reach(entry);
    names
      .filter((name) => !reached.has(name))
      .forEach((name) =>
        error(
          ["spec", "stages", name],
          "Stage is unreachable from entryStage.",
        ),
      );
    // Success contributes result outputs; failed/interrupted branches do not.
    // Intersect at joins to reject required files missing on any success path.
    if (!cyclic && reached.has(entry)) {
      type Flow = {
        must: Set<string>;
        may: Set<string>;
        artifacts: Set<string>;
      };
      const flows = new Map<string, Flow>([
        [entry, { must: new Set(), may: new Set(), artifacts: new Set() }],
      ]);
      for (const name of order) {
        const incoming = flows.get(name);
        if (!incoming) continue;
        const stage = record(stages[name]),
          p: Path = ["spec", "stages", name];
        for (const [binding, value] of Object.entries(
          record(record(stage.context).artifacts),
        )) {
          const wire = record(value),
            key = `${wire.namespace}/${wire.name}`;
          if (
            wire.required === true &&
            wire.namespace !== "inputs" &&
            !incoming.artifacts.has(key)
          ) {
            const knownProducer = names.some((producer) =>
              Object.values(
                record(record(record(stages[producer]).result).artifacts),
              ).some((slot) => {
                const from = record(record(slot).from);
                return `${from.namespace}/${from.name}` === key;
              }),
            );
            if (knownProducer)
              warning(
                [...p, "context", "artifacts", binding],
                "This file is not guaranteed to be written before this stage on every path. Check runtime/project context.",
              );
          }
        }
        const success: Flow = {
          must: new Set(incoming.must),
          may: new Set(incoming.may),
          artifacts: new Set(incoming.artifacts),
        };
        for (const [output, result] of Object.entries(
          record(stage.workflowOutputs),
        )) {
          if (incoming.may.has(output))
            error(
              [...p, "workflowOutputs", output],
              "This output can already be mapped by an earlier stage.",
            );
          success.may.add(output);
          if (
            record(record(record(stage.result).artifacts)[textValue(result)])
              .required === true
          )
            success.must.add(output);
        }
        for (const value of Object.values(
          record(record(stage.result).artifacts),
        )) {
          const slot = record(value),
            from = record(slot.from);
          if (slot.required === true && from.namespace && from.name)
            success.artifacts.add(`${from.namespace}/${from.name}`);
        }
        for (const outcome of OUTCOMES) {
          const action = continuation(record(stage.on)[outcome]),
            flow = outcome === "succeeded" ? success : incoming;
          if (action.succeed !== undefined)
            for (const [output, slot] of Object.entries(record(spec.outputs))) {
              if (record(slot).required === true && !flow.must.has(output))
                error(
                  [...p, "on", outcome],
                  `Workflow may succeed without required output ${output}.`,
                );
            }
          const target = textValue(action.next);
          if (!target || !reached.has(target)) continue;
          const previous = flows.get(target);
          flows.set(
            target,
            previous
              ? {
                  must: new Set(
                    [...previous.must].filter((key) => flow.must.has(key)),
                  ),
                  may: new Set([...previous.may, ...flow.may]),
                  artifacts: new Set(
                    [...previous.artifacts].filter((key) =>
                      flow.artifacts.has(key),
                    ),
                  ),
                }
              : {
                  must: new Set(flow.must),
                  may: new Set(flow.may),
                  artifacts: new Set(flow.artifacts),
                },
          );
        }
      }
    }
  } else if (draft.kind === "AgentTemplate") {
    requiredString(["spec", "description"]);
    selector(["spec", "runtime"]);
    selector(["spec", "sandboxProfile"]);
    if (spec.runtime !== "tool@1") {
      selector(["spec", "modelPolicy"]);
      requiredString(["spec", "instructions", "ref"]);
    } else
      for (const key of ["modelPolicy", "instructions", "summarizer", "skills"])
        if (spec[key] !== undefined)
          error(["spec", key], "tool@1 forbids this field.");
    if (spec.summarizer !== undefined) {
      selector(["spec", "summarizer", "modelPolicy"]);
      requiredString(["spec", "summarizer", "instructions", "ref"]);
    }
    for (const key of ["toolsets", "skills"]) {
      if (key === "toolsets" && spec[key] === undefined)
        error(["spec", key], "Toolsets are required; use [] for none.");
      if (spec[key] !== undefined && !Array.isArray(spec[key]))
        error(["spec", key], "Expected a list.");
      if (Array.isArray(spec[key]))
        spec[key].forEach((_, index) =>
          key === "toolsets"
            ? selector(["spec", key, index, "ref"])
            : requiredString(["spec", key, index, "name"]),
        );
    }
    const tools = new Set<string>(),
      refs = new Set<string>(),
      skills = new Set<string>();
    if (Array.isArray(spec.toolsets))
      spec.toolsets.forEach((value, index) => {
        const toolset = record(value),
          path: Path = ["spec", "toolsets", index];
        const ref = textValue(toolset.ref);
        if (refs.has(ref))
          error([...path, "ref"], "Toolset selectors must be unique.");
        refs.add(ref);
        if (!Array.isArray(toolset.tools) || !toolset.tools.length)
          error([...path, "tools"], "Select at least one tool.");
        if (Array.isArray(toolset.tools))
          for (const tool of toolset.tools) {
            if (typeof tool !== "string" || !tool.trim())
              error(
                [...path, "tools"],
                "Tool names must be non-empty strings.",
              );
            else if (tools.has(tool))
              error(
                [...path, "tools"],
                `Tool ${tool} is selected more than once.`,
              );
            tools.add(textValue(tool));
          }
      });
    if (Array.isArray(spec.skills))
      spec.skills.forEach((value, index) => {
        const skill = record(value),
          path: Path = ["spec", "skills", index];
        if (skill.namespace !== "skills" || skill.revision !== undefined)
          error(path, "Skills select namespace skills and omit revision.");
        if (skills.has(textValue(skill.name)))
          error(path, "Skill names must be unique.");
        skills.add(textValue(skill.name));
      });
  } else {
    slots(["spec", "inputs"], false);
    if (
      ![
        "operation-tracing",
        "requirements-verification",
        "custom-checklist",
        "risk-assessment",
        "finding-verification",
      ].includes(textValue(spec.mode))
    )
      error(["spec", "mode"], "Choose a supported check mode.");
    selector(["spec", "inventory", "implementation"]);
    const roles = record(spec.workflows),
      inventory = record(spec.inventory),
      dependencies = new Map<string, string[]>();
    if (!Object.keys(roles).length)
      error(["spec", "workflows"], "At least one workflow role is required.");
    if (record(roles[textValue(inventory.itemWorkflowRole)]).kind !== "check")
      error(
        ["spec", "inventory", "itemWorkflowRole"],
        "Choose a check workflow role.",
      );
    for (const [name, value] of Object.entries(roles)) {
      const role = record(value),
        p: Path = ["spec", "workflows", name],
        deps: string[] = [];
      selector([...p, "ref"]);
      if (
        !["prepare", "discovery", "check", "assessment"].includes(
          textValue(role.kind),
        )
      )
        error(
          [...p, "kind"],
          "Choose prepare, discovery, check or assessment.",
        );
      for (const key of ["inputs", "parameters", "outputs"])
        if (
          role[key] === undefined ||
          role[key] === null ||
          typeof role[key] !== "object" ||
          Array.isArray(role[key])
        )
          error([...p, key], "A mapping is required; use {} for none.");
      for (const [binding, value] of Object.entries(record(role.inputs))) {
        const row = record(value),
          path = [...p, "inputs", binding];
        if (
          ![
            "audit-input",
            "item-package",
            "execution-manifest",
            "prepare-output",
            "retained-output",
          ].includes(textValue(row.source))
        )
          error([...path, "source"], "Choose a supported input source.");
        if (
          ["item-package", "execution-manifest"].includes(
            textValue(row.source),
          ) &&
          (row.name || row.role)
        )
          error(path, "This input source forbids name and role.");
        if (row.source === "audit-input" && row.role)
          error(path, "audit-input forbids role.");
        if (
          row.source === "audit-input" &&
          !Object.hasOwn(record(spec.inputs), textValue(row.name))
        )
          error(path, "This check input is not declared.");
        if (
          ["prepare-output", "retained-output"].includes(textValue(row.source))
        ) {
          const producer = record(roles[textValue(row.role)]);
          if (!Object.hasOwn(record(producer.outputs), textValue(row.name)))
            error(path, "Producer role and logical output must exist.");
          if (row.source === "prepare-output" && producer.kind !== "prepare")
            error(path, "prepare-output must select a prepare role.");
          if (
            row.source === "retained-output" &&
            (producer.kind === "check" ||
              producer.kind === "prepare" ||
              role.kind === "prepare")
          )
            error(path, "This role cannot supply retained-output here.");
          const phases: Record<string, number> = {
            prepare: -1,
            discovery: 0,
            check: 1,
            assessment: 2,
          };
          if (
            (phases[textValue(producer.kind)] ?? 0) >
            (phases[textValue(role.kind)] ?? 0)
          )
            error(path, "Cannot depend on a later role phase.");
          deps.push(textValue(row.role));
        }
      }
      dependencies.set(name, deps);
    }
    const active = new Set<string>(),
      done = new Set<string>();
    const visitRole = (name: string) => {
      if (active.has(name)) {
        error(
          ["spec", "workflows", name, "inputs"],
          "Workflow role dependencies form a cycle.",
        );
        return;
      }
      if (done.has(name)) return;
      active.add(name);
      (dependencies.get(name) ?? []).forEach(visitRole);
      active.delete(name);
      done.add(name);
    };
    Object.keys(roles).forEach(visitRole);
    const execution = record(spec.execution);
    if (execution.roundMode !== "fixed-barrier")
      error(["spec", "execution", "roundMode"], "Use fixed-barrier.");
    for (const [key, maximum] of Object.entries({
      maxRounds: 32,
      batchSize: 64,
      maxItemsPerRound: 10000,
      maxItemsTotal: 100000,
      maxSubmittedRuns: 1000000,
      maxItemRunAttempts: 10,
      deadlineSeconds: 31536000,
      maxEvidenceBytes: 1073741824,
    }))
      if (
        !Number.isSafeInteger(execution[key]) ||
        Number(execution[key]) < 1 ||
        Number(execution[key]) > maximum
      )
        error(
          ["spec", "execution", key],
          `Use an integer between 1 and ${maximum}.`,
        );
    if (
      Number(execution.batchSize) > Number(execution.maxItemsPerRound) ||
      Number(execution.maxItemsPerRound) > Number(execution.maxItemsTotal)
    )
      error(
        ["spec", "execution"],
        "Require batchSize ≤ maxItemsPerRound ≤ maxItemsTotal.",
      );
    if (
      Number(execution.maxSubmittedRuns) <
      Math.ceil(Number(execution.maxItemsTotal) / Number(execution.batchSize))
    )
      error(
        ["spec", "execution", "maxSubmittedRuns"],
        "Run budget cannot cover the initial items.",
      );
    if (
      !["assess-with-gaps", "fail"].includes(
        textValue(execution.incompleteRound),
      )
    )
      error(
        ["spec", "execution", "incompleteRound"],
        "Choose assess-with-gaps or fail.",
      );
    for (const [key, allowed] of Object.entries({
      activeChecks: ["prohibited", "automatic", "approval-required"],
      findingConfirmation: ["human-required", "disabled"],
      notApplicable: ["human-required", "profile-rule"],
      reportAcceptance: ["automatic", "human-required"],
    }))
      if (!allowed.includes(textValue(record(spec.interaction)[key])))
        error(["spec", "interaction", key], `Choose ${allowed.join(" or ")}.`);
  }
  return [...problems, ...validateAdvanced(draft)];
}
