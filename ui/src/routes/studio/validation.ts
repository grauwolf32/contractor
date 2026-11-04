import { ARTIFACT_NAME_PATTERN, MEDIA_TYPE_PATTERN } from "../../api/artifacts";
import {
  CONFIG_NAME_PATTERN,
  CONFIG_VERSION_PATTERN,
  isConfigSelector,
} from "../../api/workflows";
import { at, record, textValue, type Draft, type Path } from "./document";
import { validateCheckMethods } from "./check-methods";
import { validateAdvanced } from "./advanced-validation";
import { plannerFacts } from "./planners";

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
function nextStages(stage: unknown): string[] {
  return OUTCOMES.flatMap((outcome) => {
    const next = continuation(record(record(stage).on)[outcome]).next;
    return typeof next === "string" ? [next] : [];
  });
}

// Configuration names, versions and selectors follow the Server grammar
// (CONFIG_NAME_PATTERN, CONFIG_VERSION_PATTERN and isConfigSelector, held to
// api/testdata/v1alpha1/config-identity-cases.json).
const NAME_MESSAGE =
  "Use a name of at most 128 lowercase letters, digits, - and _, starting with a letter.";
const VERSION_MESSAGE =
  "Use a version of at most 64 letters, digits, ., _, + and -, starting with a letter or digit.";
const SELECTOR_MESSAGE = "Use a versioned selector such as worker@1.";

/**
 * Why a file slot's media types are refused, as the Server refuses them: one
 * or more canonical type/subtype values (MEDIA_TYPE_PATTERN), each listed
 * once, or the wildcard alone.
 */
function mediaTypesProblem(value: unknown): string | undefined {
  if (
    !Array.isArray(value) ||
    !value.length ||
    value.some(
      (type) =>
        typeof type !== "string" ||
        (type !== "*/*" && !MEDIA_TYPE_PATTERN.test(type)),
    )
  )
    return "Provide one or more lowercase media types such as text/plain, or */* for any file.";
  if (new Set(value).size !== value.length) return "List each media type once.";
  if (value.includes("*/*") && value.length > 1)
    return "*/* already accepts any file; remove the other media types.";
  return undefined;
}

const COMPACT_MEMBERS = new Set([
  "strategy",
  "modelPolicy",
  "instructions",
  "includeReasoning",
  "contextWindowRatio",
  "keepRecentGroups",
  "maxCompactions",
]);

function boundedInteger(value: unknown, minimum: number, maximum: number) {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= minimum &&
    value <= maximum
  );
}

/**
 * Why an agent binding's contextManagement is refused, as the Server refuses
 * it (api/testdata/workflow-context-management-cases.json): template@1 alone,
 * or compact@1 for an isolated stage without a summarized outcome. Catalog
 * checks (policy limits, Audit completion, model-free templates) stay with the
 * Server.
 */
function contextManagementProblems(
  value: unknown,
  stage: Record<string, unknown>,
): [Path, string][] {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    return [[[], "Expected a mapping with a strategy."]];
  const context = record(value),
    strategy = context.strategy,
    problems: [Path, string][] = [];
  if (strategy === "template@1") {
    if (Object.keys(context).length !== 1)
      problems.push([[], "template@1 accepts only strategy."]);
    return problems;
  }
  if (strategy !== "compact@1")
    return [[["strategy"], "Strategy must be template@1 or compact@1."]];
  for (const name of Object.keys(context))
    if (!COMPACT_MEMBERS.has(name))
      problems.push([[name], "compact@1 does not accept this member."]);
  if (
    typeof context.modelPolicy !== "string" ||
    !isConfigSelector(context.modelPolicy)
  )
    problems.push([["modelPolicy"], SELECTOR_MESSAGE]);
  const instructions = record(context.instructions).ref;
  if (typeof instructions !== "string" || !instructions.trim())
    problems.push([
      ["instructions", "ref"],
      "Select the compaction instructions.",
    ]);
  if (typeof context.includeReasoning !== "boolean")
    problems.push([
      ["includeReasoning"],
      "Choose whether provider reasoning reaches the compactor.",
    ]);
  const ratio = context.contextWindowRatio;
  if (
    ratio !== undefined &&
    (typeof ratio !== "number" || !(ratio > 0 && ratio < 1))
  )
    problems.push([["contextWindowRatio"], "Use a ratio between 0 and 1."]);
  if (
    context.keepRecentGroups !== undefined &&
    !boundedInteger(context.keepRecentGroups, 1, 16)
  )
    problems.push([["keepRecentGroups"], "Keep 1 to 16 recent groups."]);
  if (
    context.maxCompactions !== undefined &&
    !boundedInteger(context.maxCompactions, 1, 16)
  )
    problems.push([["maxCompactions"], "Allow 1 to 16 compactions."]);
  if (stage.session === "shared")
    problems.push([[], "compact@1 supports only isolated stages."]);
  if (record(stage.on).summarized !== undefined)
    problems.push([[], "compact@1 cannot produce a summarized outcome."]);
  return problems;
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
    const value = at(draft.value, path);
    if (typeof value !== "string" || !isConfigSelector(value))
      error(path, SELECTOR_MESSAGE);
  };
  if (draft.value.apiVersion !== "contractor/v1alpha1")
    error(["apiVersion"], "Expected contractor/v1alpha1.");
  const name = at(draft.value, ["metadata", "name"]);
  if (typeof name !== "string" || !CONFIG_NAME_PATTERN.test(name))
    error(["metadata", "name"], NAME_MESSAGE);
  const version = at(draft.value, ["metadata", "version"]);
  if (typeof version !== "string" || !CONFIG_VERSION_PATTERN.test(version))
    error(["metadata", "version"], VERSION_MESSAGE);
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
        Object.hasOwn(row, "sourceScope") &&
        (draft.kind !== "Workflow" ||
          path.length !== 2 ||
          path[1] !== "inputs" ||
          name !== "rulepack" ||
          row.sourceScope !== "user" ||
          row.required !== false)
      )
        error(
          [...p, "sourceScope"],
          "Only an optional rulepack input may select the owner UserScope.",
        );
      const problem = mediaTypesProblem(row.mediaTypes);
      if (problem !== undefined) error([...p, "mediaTypes"], problem);
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
      const operation = stage.operation !== undefined;
      if (operation) {
        if (
          stage.planner !== undefined ||
          stage.instructions !== undefined ||
          stage.session === "shared" ||
          Object.keys(record(stage.agents)).length
        )
          error(
            p,
            "A server operation cannot select a planner, instructions, Workers or a shared session.",
          );
        if (
          ![
            "sitemap-openapi-join@1",
            "sitemap-apply-openapi@1",
            "openapi-prepare-work@1",
          ].includes(textValue(record(stage.operation).ref))
        )
          error(
            [...p, "operation", "ref"],
            "Select a registered server operation.",
          );
      } else {
        requiredString([...p, "objective"]);
        selector([...p, "planner"]);
      }
      if (
        stage.session !== undefined &&
        !["isolated", "shared"].includes(textValue(stage.session))
      )
        error([...p, "session"], "Session must be isolated or shared.");
      const agents = record(stage.agents),
        count = Object.keys(agents).length,
        facts = plannerFacts(textValue(stage.planner));
      if (!operation && (!count || (facts?.singleAgent && count !== 1)))
        error(
          [...p, "agents"],
          "This planner requires exactly one agent; other planners need at least one.",
        );
      for (const agent of Object.keys(agents)) {
        selector([...p, "agents", agent, "template"]);
        const context = record(agents[agent]).contextManagement;
        if (context !== undefined)
          for (const [path, message] of contextManagementProblems(
            context,
            stage,
          ))
            error(
              [...p, "agents", agent, "contextManagement", ...path],
              message,
            );
      }
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
          !operation &&
          facts?.agentResults !== false &&
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
    if (spec.modelPolicy !== undefined)
      error(
        ["spec", "modelPolicy"],
        "Choose the model policy in Workflow executionConfig.",
      );
    if (spec.runtime !== "tool@1") {
      requiredString(["spec", "instructions", "ref"]);
    } else
      for (const key of ["modelPolicy", "instructions", "summarizer", "skills"])
        if (spec[key] !== undefined)
          error(["spec", key], "tool@1 forbids this field.");
    if (spec.summarizer !== undefined) {
      if (record(spec.summarizer).modelPolicy !== undefined)
        error(
          ["spec", "summarizer", "modelPolicy"],
          "Choose the model policy in Workflow executionConfig.",
        );
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
    validateCheckMethods(spec.methods, roles, error);
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
            "round-results",
            "retained-dependency",
          ].includes(textValue(row.source))
        )
          error([...path, "source"], "Choose a supported input source.");
        if (
          ["item-package", "execution-manifest", "round-results"].includes(
            textValue(row.source),
          ) &&
          (row.name || row.role)
        )
          error(path, "This input source forbids name and role.");
        if (row.source === "retained-dependency") {
          if (role.kind !== "check")
            error(path, "retained-dependency requires a check role.");
          if (row.role || !ARTIFACT_NAME_PATTERN.test(textValue(row.name)))
            error(path, "Choose a named dependency and omit producer role.");
        }
        if (row.source === "round-results" && role.kind !== "assessment")
          error(path, "round-results requires an assessment role.");
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
      if (
        Object.values(record(role.inputs)).filter(
          (input) => record(input).source === "round-results",
        ).length > 1
      )
        error([...p, "inputs"], "Choose at most one round-results input.");
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
      activeChecks: ["prohibited", "automatic"],
      notApplicable: ["human-required", "profile-rule"],
      reportAcceptance: ["automatic", "human-required"],
    }))
      if (!allowed.includes(textValue(record(spec.interaction)[key])))
        error(["spec", "interaction", key], `Choose ${allowed.join(" or ")}.`);
  }
  return [...problems, ...validateAdvanced(draft)];
}
