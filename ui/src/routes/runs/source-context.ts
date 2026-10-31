import { createContext } from "react";
import { PROJECT_ID_PATTERN, type listProjectRuns } from "../../api/projects";
import {
  QUEUE_MEMBERSHIPS,
  QUEUE_STATES,
  type listRunQueue,
} from "../../api/queue";
import { RUN_STATES, TERMINAL_RUN_STATES, type listRuns } from "../../api/runs";
import { catalogReturnState } from "../../app/navigation";

type Polling = {
  refreshMs?: 5000 | 10000 | 15000;
  settleWhenTerminal?: boolean;
};
type RunListSource = Polling &
  (
    | {
        kind: "queue";
        label: "Queue";
        options: NonNullable<Parameters<typeof listRunQueue>[1]>;
      }
    | {
        kind: "completed";
        label: "Completed";
        options: NonNullable<Parameters<typeof listRuns>[1]>;
      }
    | {
        kind: "project";
        label: "Project Runs" | "Recent Runs" | "Successful Runs";
        projectId: string;
        options: Omit<Parameters<typeof listProjectRuns>[1], "projectId"> & {
          limit: number;
        };
      }
  );

export const RunListSourceContext = createContext<RunListSource | undefined>(
  undefined,
);

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function only(value: Record<string, unknown>, keys: readonly string[]) {
  return Object.keys(value).every((key) => keys.includes(key));
}

/** Read only context produced by the existing bounded Run-list owners. */
export function readRunListOrigin(value: unknown) {
  if (
    !record(value) ||
    !record(value.runSource) ||
    typeof value.returnTo !== "string"
  )
    return undefined;
  const source = value.runSource;
  if (
    !only(source, [
      "kind",
      "label",
      "options",
      "projectId",
      "refreshMs",
      "settleWhenTerminal",
    ]) ||
    !record(source.options)
  )
    return undefined;
  if (
    source.refreshMs !== undefined &&
    ![5000, 10000, 15000].includes(source.refreshMs as number)
  )
    return undefined;
  if (
    source.settleWhenTerminal !== undefined &&
    typeof source.settleWhenTerminal !== "boolean"
  )
    return undefined;
  const options = source.options;
  if (
    options.cursor !== undefined &&
    (typeof options.cursor !== "string" || options.cursor.length === 0)
  )
    return undefined;
  const fallback = { returnTo: "/runs", returnLabel: "Runs" };
  const origin = catalogReturnState(value, fallback);
  if (origin === fallback) return undefined;
  const pathname = origin.returnTo.split(/[?#]/, 1)[0];
  if (source.kind === "queue") {
    if (
      pathname !== "/runs" ||
      source.label !== "Queue" ||
      !only(options, ["state", "membership", "cursor"])
    )
      return undefined;
    if (
      options.state !== undefined &&
      !QUEUE_STATES.some((state) => state === options.state)
    )
      return undefined;
    if (
      options.membership !== undefined &&
      !QUEUE_MEMBERSHIPS.some((membership) => membership === options.membership)
    )
      return undefined;
  } else if (source.kind === "completed") {
    if (
      pathname !== "/runs" ||
      source.label !== "Completed" ||
      options.lifecycle !== "terminal" ||
      !only(options, ["state", "lifecycle", "cursor", "labelSelectors"])
    )
      return undefined;
    if (
      options.state !== undefined &&
      !TERMINAL_RUN_STATES.some((state) => state === options.state)
    )
      return undefined;
    if (
      options.labelSelectors !== undefined &&
      (!Array.isArray(options.labelSelectors) ||
        options.labelSelectors.some(
          (selector) =>
            !record(selector) ||
            !only(selector, ["key", "value"]) ||
            typeof selector.key !== "string" ||
            typeof selector.value !== "string",
        ))
    )
      return undefined;
  } else if (source.kind === "project") {
    if (
      typeof source.projectId !== "string" ||
      !PROJECT_ID_PATTERN.test(source.projectId) ||
      !["Project Runs", "Recent Runs", "Successful Runs"].includes(
        source.label as string,
      )
    )
      return undefined;
    const project = `/projects/${encodeURIComponent(source.projectId)}`;
    if (pathname !== project && pathname !== `${project}/runs`)
      return undefined;
    if (
      !only(options, ["limit", "lifecycle", "state", "cursor"]) ||
      ![3, 5, 25].includes(options.limit as number)
    )
      return undefined;
    if (
      options.state !== undefined &&
      !RUN_STATES.some((state) => state === options.state)
    )
      return undefined;
    if (
      options.lifecycle !== undefined &&
      !["active", "terminal"].includes(options.lifecycle as string)
    )
      return undefined;
  } else return undefined;
  return { ...origin, runSource: source as RunListSource };
}

export type RunListOrigin = NonNullable<ReturnType<typeof readRunListOrigin>>;
