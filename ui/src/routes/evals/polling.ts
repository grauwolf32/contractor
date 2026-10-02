import { EVAL_POLL_MS, type EvalExperiment } from "../../api/evals";

export const EVAL_SETTLED_POLL_MS = 30_000;

type EvalPollState = Pick<
  EvalExperiment,
  "state" | "freshness" | "deletionRequestedAt"
>;
type EvalListPollState = Pick<EvalExperiment, "state" | "freshness">;

function isTerminalAndCurrent(value: EvalListPollState): boolean {
  return (
    (value.state === "finished" || value.state === "cancelled") &&
    value.freshness === "current"
  );
}

export function evalExperimentPollInterval(
  value: EvalPollState | undefined,
): number | false {
  if (value?.state === "draft") return false;
  if (
    value !== undefined &&
    isTerminalAndCurrent(value) &&
    value.deletionRequestedAt == null
  ) {
    return EVAL_SETTLED_POLL_MS;
  }
  return EVAL_POLL_MS;
}

export function evalListPollInterval(
  items: EvalListPollState[] | undefined,
  paged: boolean,
): number | false {
  if (paged) return false;
  if (items !== undefined && items.every(isTerminalAndCurrent)) {
    return EVAL_SETTLED_POLL_MS;
  }
  return EVAL_POLL_MS;
}
