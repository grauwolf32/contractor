import { CONFIG_ID_PATTERN, isConfigSelector } from "../../api/workflows";
import { record, textValue, type Path } from "./document";

export const FINDING_CHECK_PLAN_SCHEMA =
  "contractor.audit.finding-check-plan.v1";
export const MAX_CHECK_METHODS = 128;
export const MAX_CHECK_CAPABILITIES = 32;
const MAX_CHECK_CAPABILITY_TOOLS = 128;
export const CHECK_METHODS: readonly string[] = [
  "verify",
  "replay",
  "applicability",
  "alternative",
  "static-trace",
  "http-probe",
];

export function validateCheckMethods(
  methods: unknown,
  workflows: Record<string, unknown>,
  error: (path: Path, message: string) => void,
): void {
  if (methods === undefined) return;
  if (!methods || typeof methods !== "object" || Array.isArray(methods)) {
    error(["spec", "methods"], "Methods must be a mapping.");
    return;
  }
  const entries = Object.entries(methods),
    selected = new Set<string>();
  if (entries.length > MAX_CHECK_METHODS)
    error(["spec", "methods"], "Choose at most 128 method mappings.");
  for (const [name, input] of entries) {
    const value = record(input),
      path: Path = ["spec", "methods", name];
    const fields = [
      "planSchema",
      "method",
      "workflowRole",
      "requiredCapabilities",
    ];
    if (
      Object.keys(value).some((key) => !fields.includes(key)) ||
      fields.some((key) => !Object.hasOwn(value, key))
    )
      error(path, "Use the complete registered method mapping.");
    if (
      /^\p{White_Space}*$/u.test(name) ||
      name.includes("\0") ||
      name.includes("/") ||
      new TextEncoder().encode(name).length > 128
    )
      error(path, "Use a bounded method mapping name without slash.");
    const method = textValue(value.method);
    if (
      value.planSchema !== FINDING_CHECK_PLAN_SCHEMA ||
      !CHECK_METHODS.includes(method)
    )
      error(path, "Choose a registered plan method.");
    const key = `${textValue(value.planSchema)}\0${method}`;
    if (selected.has(key))
      error(path, "Each schema and method selects one route.");
    selected.add(key);
    if (record(workflows[textValue(value.workflowRole)]).kind !== "check")
      error([...path, "workflowRole"], "Choose a check workflow role.");
    if (
      !Array.isArray(value.requiredCapabilities) ||
      value.requiredCapabilities.length > MAX_CHECK_CAPABILITIES
    ) {
      error(
        [...path, "requiredCapabilities"],
        "Provide at most 32 required tool selections.",
      );
      continue;
    }
    const capabilities = new Set<string>();
    value.requiredCapabilities.forEach((raw, index) => {
      const capability = record(raw),
        ref = record(capability.ref),
        p = [...path, "requiredCapabilities", index];
      if (
        Object.keys(capability).length !== 4 ||
        !["stage", "agent", "ref", "tools"].every((key) =>
          Object.hasOwn(capability, key),
        ) ||
        Object.keys(ref).length !== 2 ||
        !Object.hasOwn(ref, "toolsetId") ||
        !Object.hasOwn(ref, "version")
      )
        error(p, "Use an exact stage, worker, toolset and tools selection.");
      if (
        !CONFIG_ID_PATTERN.test(textValue(capability.stage)) ||
        !CONFIG_ID_PATTERN.test(textValue(capability.agent)) ||
        !isConfigSelector(
          `${textValue(ref.toolsetId)}@${textValue(ref.version)}`,
        )
      )
        error(
          p,
          "Required tools need exact stage, worker and toolset identities.",
        );
      const identity = `${textValue(capability.stage)}\0${textValue(capability.agent)}\0${textValue(ref.toolsetId)}@${textValue(ref.version)}`;
      if (capabilities.has(identity))
        error(p, "Each required Worker toolset appears once.");
      capabilities.add(identity);
      const tools = capability.tools;
      if (
        !Array.isArray(tools) ||
        tools.length < 1 ||
        tools.length > MAX_CHECK_CAPABILITY_TOOLS ||
        tools.some(
          (tool) =>
            typeof tool !== "string" ||
            /^\p{White_Space}*$/u.test(tool) ||
            tool.includes("\0"),
        ) ||
        new Set(tools).size !== tools.length
      )
        error([...p, "tools"], "Choose nonempty unique tools, up to 128.");
    });
  }
}
