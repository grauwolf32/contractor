import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { listProjectRuns } from "../../api/projects";
import type { RunSummary } from "../../api/runs";
import { queryKeys } from "../../api/query-keys";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "../runs/components";
import { ProjectRegion } from "./common";
import { groupEvaluationRuns } from "./evaluation-groups";

function EvaluationRuns({ runs }: { runs: readonly RunSummary[] }) {
  return (
    <div className="eval-run-groups">
      {groupEvaluationRuns(runs).map((group) => (
        <section className="eval-run-group" key={group.id}>
          <div className="eval-run-group-heading">
            <div>
              <p className="eyebrow">eval.id</p>
              <h4>
                {group.id === "" ? (
                  "Runs without eval.id"
                ) : (
                  <code>{group.id}</code>
                )}
              </h4>
            </div>
            <span className="project-workflow-count">
              {group.runs.length} {group.runs.length === 1 ? "Run" : "Runs"}
            </span>
          </div>
          {group.names.length === 0 ? null : (
            <p className="muted-copy">
              eval.name: <code>{group.names.join(", ")}</code>
            </p>
          )}
          <div className="table-scroll">
            <table className="responsive-table eval-run-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Leg</th>
                  <th>Case</th>
                  <th>Sample</th>
                  <th>Workflow</th>
                  <th>State</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                {group.runs.map((run) => (
                  <tr key={run.runId}>
                    <td data-label="Run">
                      <Link to={`/runs/${encodeURIComponent(run.runId)}`}>
                        {run.runId}
                      </Link>
                    </td>
                    <td data-label="Leg">
                      <code>{run.labels["eval.leg"] ?? "—"}</code>
                    </td>
                    <td data-label="Case">
                      <span className="eval-case-value">
                        {run.labels["eval.fixture"] === undefined ? null : (
                          <small>{run.labels["eval.fixture"]}</small>
                        )}
                        <code>{run.labels["eval.case"] ?? "—"}</code>
                      </span>
                    </td>
                    <td data-label="Sample">
                      <code>{run.labels["eval.sample"] ?? "—"}</code>
                    </td>
                    <td data-label="Workflow">
                      <code>{run.workflow}</code>
                    </td>
                    <td data-label="State">
                      <StateBadge state={run.state} />
                    </td>
                    <td data-label="Updated">
                      {formatTimestamp(run.updatedAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}

export function ProjectRunsRegion({
  projectId,
  evaluation,
}: {
  projectId: string;
  evaluation: boolean;
}) {
  const api = usePublicAPI();
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const query = useQuery({
    queryKey: queryKeys.projects.runs(projectId, cursor),
    queryFn: () =>
      listProjectRuns(api, {
        projectId,
        ...(cursor === undefined ? {} : { cursor }),
      }),
  });

  return (
    <ProjectRegion
      eyebrow="Execution history"
      title={evaluation ? "Eval Runs" : "Project Runs"}
      id="project-runs"
      action={<Link to="/runs">All Runs →</Link>}
    >
      {query.isPending ? (
        <p className="loading-copy" aria-live="polite">
          Loading {evaluation ? "Eval" : "Project"} Runs…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">
          <strong>
            No Workflow Runs belong to this {evaluation ? "Eval" : "Project"}.
          </strong>
          <p>Launch one compatible Workflow when inputs are ready.</p>
        </div>
      ) : evaluation ? (
        <EvaluationRuns runs={query.data.items} />
      ) : (
        <div className="table-scroll">
          <table className="responsive-table project-run-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Workflow</th>
                <th>State</th>
                <th>Labels</th>
                <th>Updated</th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((run) => (
                <tr key={run.runId}>
                  <td data-label="Run">
                    <Link to={`/runs/${encodeURIComponent(run.runId)}`}>
                      {run.runId}
                    </Link>
                  </td>
                  <td data-label="Workflow">
                    <code>{run.workflow}</code>
                  </td>
                  <td data-label="State">
                    <StateBadge state={run.state} />
                  </td>
                  <td data-label="Labels">
                    <RunMetadataLabelChips labels={run.labels} />
                  </td>
                  <td data-label="Updated">{formatTimestamp(run.updatedAt)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CursorControls
        label="Project Run pages"
        canGoBack={cursors.length > 1}
        {...(query.data?.page.hasMore === true &&
        query.data.page.nextCursor !== undefined
          ? { nextCursor: query.data.page.nextCursor }
          : {})}
        onBack={() =>
          setCursors((current) =>
            current.slice(0, Math.max(1, current.length - 1)),
          )
        }
        onNext={(next) => setCursors((current) => [...current, next])}
      />
    </ProjectRegion>
  );
}
