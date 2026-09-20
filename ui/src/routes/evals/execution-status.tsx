import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { listEvalMembers, type EvalExperiment } from "../../api/evals";
import { EvalError } from "./common";

export function EvalExecutionStatus({
  experiment,
}: {
  experiment: EvalExperiment;
}) {
  const counts = Object.values(experiment.summary?.counts ?? {});
  const total = (key: keyof (typeof counts)[number]) =>
    counts.reduce((sum, arm) => sum + arm[key], 0);
  const submitted = total("submitted");
  const terminal = total("terminal");
  const failed = terminal - total("executionSucceeded");
  const excluded = total("unsupported") + total("blocked");
  const finished = ["finished", "cancelled", "interrupted"].includes(
    experiment.state,
  );
  const base = `/evals/experiments/${encodeURIComponent(experiment.experimentId)}`;

  if (experiment.state === "paused" || experiment.state === "pausing") {
    return (
      <section className="panel eval-panel" aria-label="Execution status">
        <h2>
          {experiment.state === "paused"
            ? "Dispatch paused"
            : "Waiting for active executions"}
        </h2>
        <p>
          {experiment.state === "paused"
            ? "Use Resume to continue this experiment. Its original deadline still applies."
            : "Accepted executions are draining. Resume becomes available when they finish."}
        </p>
      </section>
    );
  }
  if (!finished && !(experiment.state === "ready" && excluded > 0)) return null;

  let title = "Execution finished";
  if (experiment.state === "ready") title = "Some attempts cannot be started";
  else if (submitted === 0) title = "No executions started";
  else if (terminal < submitted) title = "Execution evidence is incomplete";
  else if (failed > 0) title = "Some executions ended without success";
  else if (total("scored") < terminal)
    title = "Execution finished; assessment is incomplete";

  let reasonFilter: "unsupported" | "blocked" | "failed" = "failed";
  if (total("blocked") > 0) reasonFilter = "blocked";
  if (total("unsupported") > 0) reasonFilter = "unsupported";
  return (
    <section className="panel eval-panel" aria-label="Execution status">
      <h2>{title}</h2>
      <p>
        {submitted} / {total("expected")} submitted · {terminal} finished ·{" "}
        {total("scored")} scored.
        {excluded > 0
          ? ` Preparation: ${total("unsupported")} unsupported, ${total("blocked")} blocked.`
          : ""}
        {failed > 0 ? ` ${failed} executions did not succeed.` : ""}
      </p>
      {excluded > 0 || failed > 0 ? (
        <ExecutionReasons experiment={experiment} filter={reasonFilter} />
      ) : null}
      {total("collectionComplete") > total("scored") ? (
        <p>
          Collected results still need assessment.{" "}
          <Link to={`${base}/comparison?filter=unresolved`}>
            Review results in Comparison
          </Link>
          . Completing their assessment does not require another execution.
        </p>
      ) : null}
      {finished &&
      experiment.controlMode === "server" &&
      experiment.allowedCommands.includes("duplicate") ? (
        <p>
          This attempt is closed. To run again, choose{" "}
          <strong>Duplicate</strong>, review the copied setup, then{" "}
          <strong>Prepare</strong> and <strong>Start</strong>. Preparation
          checks compatibility again; the original results stay available.
        </p>
      ) : null}
      <Link to={`${base}/attempts?filter=all`}>
        Inspect all attempts and execution evidence
      </Link>
    </section>
  );
}

function ExecutionReasons({
  experiment,
  filter,
}: {
  experiment: EvalExperiment;
  filter: "unsupported" | "blocked" | "failed";
}) {
  const api = usePublicAPI();
  const query = {
    filter,
    ...(experiment.viewSnapshot
      ? { viewSnapshot: experiment.viewSnapshot }
      : {}),
  };
  const members = useQuery({
    queryKey: ["evals", "status-reasons", experiment.experimentId, query],
    queryFn: ({ signal }) =>
      listEvalMembers(api, experiment.experimentId, query, signal),
  });
  const reasons = [
    ...new Set(
      (members.data?.items ?? []).flatMap((member) =>
        [member.member.reason, member.execution?.reason].filter(
          (reason): reason is string => !!reason,
        ),
      ),
    ),
  ];
  return (
    <>
      <EvalError error={members.error} reload={() => void members.refetch()} />
      {reasons.length ? (
        <ul>
          {reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      ) : null}
      {members.data?.page.hasMore ? (
        <p>More affected attempts are listed under Attempts.</p>
      ) : null}
    </>
  );
}
