import type { components } from "./generated/public";

type Requirement = components["schemas"]["ToolsetConfigurationRequirement"];
type Coverage = components["schemas"]["ToolsetConfigurationCoverage"];
type Section = NonNullable<
  components["schemas"]["RuntimeLabelBinding"]["changedSections"]
>[number];

const REF = /^[a-z][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9._+-]*$/;
const NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const TOOL = /^[A-Za-z_][A-Za-z0-9_]{0,63}$/;
const DIGEST = /^sha256:[0-9a-f]{64}$/;
const SECTIONS: readonly Section[] = [
  "planner.telemetry",
  "worker.caido",
  "worker.httpProxy",
  "worker.llmGateway",
  "worker.telemetry",
  "worker.toolsets",
];

function invalid(): never {
  throw new TypeError("Toolset configuration projection is invalid");
}

function record(
  value: unknown,
  keys: readonly string[],
): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    return invalid();
  const object = value as Record<string, unknown>;
  if (
    Object.keys(object).length !== keys.length ||
    keys.some((key) => !(key in object))
  )
    return invalid();
  return object;
}

function array(value: unknown, max: number, min = 0): unknown[] {
  if (!Array.isArray(value) || value.length < min || value.length > max)
    return invalid();
  return value;
}

function text(value: unknown, pattern: RegExp, max = 256): string {
  if (typeof value !== "string" || value.length > max || !pattern.test(value))
    return invalid();
  return value;
}

function sorted<T>(values: T[], key: (item: T) => string): T[] {
  if (
    values.some(
      (item, index) => index > 0 && key(values[index - 1]!) >= key(item),
    )
  )
    return invalid();
  return values;
}

export function safeToolsetConfigurationRequirements(
  value: unknown,
): Requirement[] {
  return sorted(
    array(value, 32).map((candidate): Requirement => {
      const row = record(candidate, [
        "ref",
        "configurationRequired",
        "configurationSchema",
        "toolDiscovery",
        "workerRuntimes",
        "tools",
        "consumers",
      ]);
      if (
        typeof row.configurationRequired !== "boolean" ||
        (row.toolDiscovery !== "static" && row.toolDiscovery !== "allocation")
      )
        return invalid();
      const workerRuntimes = sorted(
        array(row.workerRuntimes, 32, 1).map((value) => text(value, REF)),
        (value) => value,
      );
      const tools = sorted(
        array(row.tools, 1024, 1).map((value) => text(value, TOOL)),
        (value) => value,
      );
      const consumers = sorted(
        array(row.consumers, 256, 1).map((value) => {
          const consumer = record(value, ["stage", "agent"]);
          return {
            stage: text(consumer.stage, NAME),
            agent: text(consumer.agent, NAME),
          };
        }),
        (value) => `${value.stage}\0${value.agent}`,
      );
      return {
        ref: text(row.ref, REF),
        configurationRequired: row.configurationRequired,
        configurationSchema: text(row.configurationSchema, REF),
        toolDiscovery: row.toolDiscovery,
        workerRuntimes,
        tools,
        consumers,
      };
    }),
    (value) => value.ref,
  );
}

export function safeToolsetConfigurationCoverage(value: unknown): Coverage[] {
  return sorted(
    array(value, 32).map((candidate): Coverage => {
      if (
        typeof candidate !== "object" ||
        candidate === null ||
        !("state" in candidate)
      )
        return invalid();
      if (candidate.state === "configured") {
        const row = record(candidate, ["ref", "state", "valueDigest"]);
        return {
          ref: text(row.ref, REF),
          state: "configured",
          valueDigest: text(row.valueDigest, DIGEST),
        };
      }
      if (candidate.state === "cleared") {
        const row = record(candidate, ["ref", "state"]);
        return { ref: text(row.ref, REF), state: "cleared" };
      }
      return invalid();
    }),
    (value) => value.ref,
  );
}

export function safeRuntimeChangedSections(value: unknown): Section[] {
  return sorted(
    array(value, 6).map((item) => {
      if (!SECTIONS.includes(item as Section)) return invalid();
      return item as Section;
    }),
    (value) => value,
  );
}
