import {
  EVAL_POLL_MS,
  isTerminalEvalState,
  type EvalExperiment,
} from "../../api/evals";
import { pollInterval } from "../../app/polling";

// A settled view still changes on its own (deletion, a refreshed snapshot), so
// it keeps a slow poll instead of stopping.
const EVAL_SETTLED_POLL_MS = 30_000;

type EvalProgress = Pick<EvalExperiment, "state" | "freshness">;

function terminalAndCurrent(value: EvalProgress): boolean {
  return isTerminalEvalState(value.state) && value.freshness === "current";
}

const experimentPoll = pollInterval<
  EvalProgress & Pick<EvalExperiment, "deletionRequestedAt">
>({
  interval: EVAL_POLL_MS,
  settled: (experiment) =>
    terminalAndCurrent(experiment) && experiment.deletionRequestedAt == null,
  settledInterval: EVAL_SETTLED_POLL_MS,
});

/**
 * One experiment: short polling while it can change, slow once it is
 * terminal with a current view, none for a draft (only its author changes
 * it), and backing off while reads fail.
 */
export const evalExperimentPollInterval: typeof experimentPoll = (query) =>
  query.state.data?.state === "draft" ? false : experimentPoll(query);

/**
 * The first experiments page: slow once every row is terminal and current,
 * short while any can change, backing off while reads fail. Later pages do
 * not poll.
 */
export const evalListPollInterval = pollInterval<{ items: EvalProgress[] }>({
  interval: EVAL_POLL_MS,
  settled: (page) => page.items.every(terminalAndCurrent),
  settledInterval: EVAL_SETTLED_POLL_MS,
});
