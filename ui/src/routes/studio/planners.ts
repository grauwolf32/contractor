/**
 * The Studio's view of the Server's PlannerFactory descriptors (CAT-R79).
 * planners.test.ts holds it to api/testdata/planner-fact-cases.json, which
 * the Server's descriptor registry test reads too.
 */
interface PlannerFacts {
  /** The Stage binds exactly one Agent; every other Planner takes one or more. */
  singleAgent: boolean;
  /** Each Stage result comes from a namespace of one of the Stage's Agents. */
  agentResults: boolean;
}

const PLANNER_FACTS: Readonly<Record<string, PlannerFacts>> = {
  "passthrough@1": { singleAgent: true, agentResults: true },
  "router@1": { singleAgent: false, agentResults: true },
  "scan-plan@1": { singleAgent: false, agentResults: false },
  "streamline@1": { singleAgent: true, agentResults: true },
};

/** The facts of a registered Planner, or undefined for any other selector. */
export function plannerFacts(planner: string): PlannerFacts | undefined {
  return Object.hasOwn(PLANNER_FACTS, planner)
    ? PLANNER_FACTS[planner]
    : undefined;
}
