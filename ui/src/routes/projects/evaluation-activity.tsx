import { useQuery } from "@tanstack/react-query";
import { ContextLink } from "../../app/context-navigation";
import { RecordedTime } from "../../app/recorded-time";
import { usePublicAPI } from "../../api/context";
import { listProjectRuns } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { StateBadge } from "../runs/components";

export function EvaluationActivity({ projectId }: { projectId: string }) {
  const api = usePublicAPI();
  const query = useQuery({
    queryKey: queryKeys.projects.runView(projectId, { limit: 1 }),
    queryFn: () => listProjectRuns(api, { projectId, limit: 1 }),
    staleTime: 30000,
    retry: false,
  });
  const run = query.data?.items[0];
  return (
    <div className="eval-card-activity">
      {query.isPending ? (
        <small>Loading last execution…</small>
      ) : query.error ? (
        <small>Last execution unavailable. Open the workspace to retry.</small>
      ) : run === undefined ? (
        <small>No executions yet.</small>
      ) : (
        <>
          <div>
            <span>Last execution</span>
            <StateBadge state={run.state} />
          </div>
          <ContextLink
            to={`/runs/${encodeURIComponent(run.runId)}`}
            returnLabel="Eval workspaces"
          >
            {run.workflow} →
          </ContextLink>
          <span className="eval-card-labels">
            {["eval.case", "eval.leg", "eval.sample"]
              .flatMap((key) =>
                run.labels[key] === undefined
                  ? []
                  : [`${key.slice(5)}: ${run.labels[key]}`],
              )
              .join(" · ")}
          </span>
          <RecordedTime value={run.finishedAt ?? run.updatedAt} />
        </>
      )}
    </div>
  );
}
