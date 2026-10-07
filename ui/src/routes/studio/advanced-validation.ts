import { record, textValue, type Draft, type Path } from "./document";
import type { Problem } from "./validation";

const selector = /^[A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.+-]*$/;
const toolName = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
const bytes = (value: string) => new TextEncoder().encode(value).length;
const mapping = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const bindingName = (value: unknown): value is string =>
  typeof value === "string" &&
  bytes(value) > 0 &&
  bytes(value) <= 128 &&
  value.trim() === value &&
  ![...value].some((char) => [0, 9, 10, 13].includes(char.charCodeAt(0)));

/** Authored checks only; installed templates, tools and credentials require CLI resolution. */
export function validateAdvanced(draft: Draft): Problem[] {
  const problems: Problem[] = [];
  const error = (path: Path, message: string) =>
    problems.push({ path, message, severity: "error" });
  const map = (value: unknown, path: Path) => {
    if (!mapping(value)) {
      error(path, "Expected a mapping.");
      return false;
    }
    return true;
  };
  const selection = (value: unknown, path: Path, allowClear: boolean) => {
    if (!map(value, path)) return;
    const row = record(value);
    if (
      !["modelPolicy", "llmGateway", "credential"].some((key) =>
        Object.hasOwn(row, key),
      )
    )
      error(path, "Select at least one execution field.");
    for (const key of ["modelPolicy", "llmGateway"])
      if (
        Object.hasOwn(row, key) &&
        (typeof row[key] !== "string" || !selector.test(textValue(row[key])))
      )
        error([...path, key], "Use a non-null versioned selector.");
    if (
      Object.hasOwn(row, "credential") &&
      !(allowClear && row.credential === null) &&
      (typeof row.credential !== "string" ||
        !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(row.credential))
    )
      error(
        [...path, "credential"],
        allowClear
          ? "Use a credential ID, omit to inherit, or null to clear."
          : "Use a credential ID or omit to inherit; null is escalation-only.",
      );
  };
  const stageSelections = (
    value: unknown,
    path: Path,
    agents: Record<string, unknown>,
    allowClear: boolean,
  ) => {
    if (!map(value, path)) return;
    const row = record(value);
    if (row.planner !== undefined)
      selection(row.planner, [...path, "planner"], allowClear);
    if (Object.hasOwn(row, "planner") && row.planner === undefined)
      error([...path, "planner"], "Expected a mapping.");
    if (Object.hasOwn(row, "agents") && map(row.agents, [...path, "agents"])) {
      if (allowClear && !Object.keys(record(row.agents)).length)
        error(
          [...path, "agents"],
          "Inline escalation agents must not be empty.",
        );
      for (const [name, value] of Object.entries(record(row.agents))) {
        if (!Object.hasOwn(agents, name))
          error(
            [...path, "agents", name],
            "This logical agent does not exist on the stage.",
          );
        selection(value, [...path, "agents", name], allowClear);
      }
    }
  };
  const spec = record(draft.value.spec);
  if (draft.kind === "Workflow") {
    const stages = record(spec.stages),
      configPath = ["spec", "executionConfig"];
    if (
      Object.hasOwn(spec, "executionConfig") &&
      map(spec.executionConfig, configPath)
    ) {
      const config = record(spec.executionConfig);
      for (const key of ["planner", "workers"])
        if (Object.hasOwn(config, key))
          selection(config[key], [...configPath, key], false);
      if (
        Object.hasOwn(config, "stages") &&
        map(config.stages, [...configPath, "stages"])
      )
        for (const [name, value] of Object.entries(record(config.stages))) {
          if (!Object.hasOwn(stages, name))
            error(
              [...configPath, "stages", name],
              "This stage does not exist.",
            );
          stageSelections(
            value,
            [...configPath, "stages", name],
            record(record(stages[name]).agents),
            false,
          );
        }
    }
    for (const [name, value] of Object.entries(stages)) {
      const stage = record(value),
        p: Path = ["spec", "stages", name],
        context = record(stage.context),
        workspace = context.workspace;
      if (Object.hasOwn(stage, "executionConfig"))
        error(
          [...p, "executionConfig"],
          "Put stage overrides under spec.executionConfig.stages.<stage>.",
        );
      for (const outcome of ["failed", "interrupted"]) {
        const action = record(record(record(stage.on)[outcome]).escalate);
        if (!Object.hasOwn(action, "executionConfig")) continue;
        const path = [...p, "on", outcome, "escalate", "executionConfig"],
          config = record(action.executionConfig);
        if (!map(action.executionConfig, path)) continue;
        const ref = Object.hasOwn(config, "ref"),
          inline =
            Object.hasOwn(config, "planner") || Object.hasOwn(config, "agents");
        if (ref === inline)
          error(
            path,
            "Choose exactly a published ref or inline planner/agents.",
          );
        if (
          ref &&
          (typeof config.ref !== "string" || !selector.test(config.ref))
        )
          error(
            [...path, "ref"],
            "Use a versioned execution configuration selector.",
          );
        if (inline) stageSelections(config, path, record(stage.agents), true);
      }
      if (
        !Object.hasOwn(context, "workspace") ||
        !map(workspace, [...p, "context", "workspace"])
      )
        continue;
      const w = record(workspace),
        path = [...p, "context", "workspace"],
        inputs = record(context.artifacts),
        targets: string[] = [];
      if (!["direct", "overlay"].includes(textValue(w.mode)))
        error([...path, "mode"], "Workspace mode must be direct or overlay.");
      if (
        !Array.isArray(w.sources) ||
        !w.sources.length ||
        w.sources.length > 32
      )
        error(
          [...path, "sources"],
          "Provide between 1 and 32 workspace sources.",
        );
      if (Array.isArray(w.sources))
        w.sources.slice(0, 32).forEach((source, index) => {
          const s = record(source),
            sp = [...path, "sources", index];
          if (!map(source, sp)) return;
          if (
            typeof s.artifact !== "string" ||
            record(inputs[s.artifact]).required !== true
          )
            error([...sp, "artifact"], "Choose a required stage input alias.");
          const target = s.target;
          if (
            typeof target !== "string" ||
            (target !== "" &&
              (target !== target.normalize("NFC") ||
                bytes(target) > 1024 ||
                target.startsWith("/") ||
                target.includes("\\") ||
                [...target].some(
                  (char) =>
                    char.charCodeAt(0) < 32 || char.charCodeAt(0) === 127,
                ) ||
                target.includes("://") ||
                /^[A-Za-z]:/.test(target) ||
                target.split("/").length > 32 ||
                target
                  .split("/")
                  .some((part) => ["", ".", ".."].includes(part))))
          )
            error(
              [...sp, "target"],
              "Use a normalized relative POSIX directory, or an empty root target.",
            );
          if (typeof target === "string") {
            if (
              targets.some(
                (other) =>
                  target === other ||
                  target === "" ||
                  other === "" ||
                  target.startsWith(`${other}/`) ||
                  other.startsWith(`${target}/`),
              )
            )
              error(
                [...sp, "target"],
                "Workspace source targets must be unique and non-overlapping.",
              );
            targets.push(target);
          }
        });
      if (Object.hasOwn(w, "state") && map(w.state, [...path, "state"])) {
        const alias = record(w.state).artifact;
        if (typeof alias !== "string" || !Object.hasOwn(inputs, alias))
          error(
            [...path, "state", "artifact"],
            "Choose a declared stage input alias.",
          );
      }
      if (Object.hasOwn(w, "export") && map(w.export, [...path, "export"])) {
        const exports = record(w.export),
          results = record(record(stage.result).artifacts);
        if (w.mode !== "overlay")
          error([...path, "export"], "Workspace export requires overlay mode.");
        if (exports.state === exports.diff)
          error(
            [...path, "export"],
            "Workspace state and diff outputs must be distinct.",
          );
        for (const [key, media] of [
          ["state", "application/vnd.contractor.workspace-overlay+json"],
          ["diff", "text/x-diff"],
        ]) {
          const name = exports[key!],
            slot = typeof name === "string" ? record(results[name]) : {};
          if (
            typeof name !== "string" ||
            !Object.hasOwn(results, name) ||
            !Array.isArray(slot.mediaTypes) ||
            slot.mediaTypes.length !== 1 ||
            slot.mediaTypes[0] !== media ||
            Object.hasOwn(slot, "from")
          )
            error(
              [...path, "export", key!],
              `Choose a runtime-owned result slot with only ${media} and no producer binding.`,
            );
        }
      }
    }
  } else if (draft.kind === "AgentTemplate") {
    if (spec.runtime !== "tool@1") {
      if (Object.hasOwn(spec, "execution"))
        error(
          ["spec", "execution"],
          "Tool execution is only supported by tool@1.",
        );
    } else {
      const path = ["spec", "execution"],
        row = record(spec.execution);
      if (map(spec.execution, path)) {
        if (!toolName.test(textValue(row.tool)))
          error(
            [...path, "tool"],
            "Use a tool identifier of up to 64 characters.",
          );
        if (
          !bindingName(row.resultArtifact) ||
          textValue(row.resultArtifact).startsWith("tool-invocation.")
        )
          error(
            [...path, "resultArtifact"],
            "Use a result binding name outside the reserved tool-invocation namespace.",
          );
        if (
          !Number.isInteger(row.timeoutSeconds) ||
          Number(row.timeoutSeconds) < 1 ||
          Number(row.timeoutSeconds) > 3600
        )
          error(
            [...path, "timeoutSeconds"],
            "Timeout must be an integer from 1 to 3600 seconds.",
          );
        if (map(row.arguments, [...path, "arguments"])) {
          if (Object.keys(record(row.arguments)).length > 32)
            error([...path, "arguments"], "At most 32 arguments are allowed.");
          for (const [name, value] of Object.entries(record(row.arguments))) {
            const p = [...path, "arguments", name],
              arg = record(value);
            if (!toolName.test(name))
              error(p, "Use an argument identifier of up to 64 characters.");
            if (!map(value, p)) continue;
            const literal = arg.source === "literal",
              keys = Object.keys(arg);
            if (
              keys.length !== 2 ||
              !keys.includes("source") ||
              !keys.includes(literal ? "value" : "name")
            )
              error(p, "Each argument needs exactly a source and its payload.");
            if (literal) {
              const value = arg.value;
              if (!(
                typeof value === "boolean" ||
                (typeof value === "string" && bytes(value) <= 8192) ||
                (typeof value === "number" &&
                  Number.isFinite(value) &&
                  Math.abs(value) <= Number.MAX_SAFE_INTEGER)
              ))
                error(
                  [...p, "value"],
                  "Use a bounded non-null string, number or boolean literal.",
                );
            } else if (
              !["parameter", "artifact"].includes(textValue(arg.source))
            )
              error([...p, "source"], "Choose parameter, artifact or literal.");
            else if (!bindingName(arg.name))
              error(
                [...p, "name"],
                "Use a non-empty binding name of up to 128 bytes without surrounding whitespace or control characters.",
              );
          }
        }
      }
      const toolsets = spec.toolsets;
      if (
        spec.sandboxProfile !== "local-workdir@1" ||
        !Array.isArray(toolsets) ||
        toolsets.length !== 1 ||
        !Array.isArray(record(toolsets[0]).tools) ||
        (record(toolsets[0]).tools as unknown[]).length !== 1 ||
        (record(toolsets[0]).tools as unknown[])[0] !== row.tool
      )
        error(
          ["spec", "toolsets"],
          "tool@1 requires local-workdir@1 and one selected tool matching execution.tool.",
        );
    }
    if (Object.hasOwn(record(spec.summarizer), "contextWindowRatio")) {
      const ratio = record(spec.summarizer).contextWindowRatio;
      if (
        typeof ratio !== "number" ||
        !Number.isFinite(ratio) ||
        ratio <= 0 ||
        ratio >= 1
      )
        error(
          ["spec", "summarizer", "contextWindowRatio"],
          "Context window ratio must be greater than 0 and less than 1.",
        );
    }
  }
  return problems;
}
