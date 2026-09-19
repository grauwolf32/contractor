import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { usePublicAPI } from "../../api/context";
import { listProjectAudits } from "../../api/audits";
import { listProjectArtifacts } from "../../api/project-artifacts";
import { listProjectRuns, type Project } from "../../api/projects";
import { queryKeys } from "../../api/query-keys";
import { getRun, type RunSummary } from "../../api/runs";
import { getWorkflow } from "../../api/workflows";
import { ContextLink } from "../../app/context-navigation";
import { ErrorNotice, formatTimestamp } from "../artifacts/common";
import { StateBadge } from "../runs/components";
import {
  organizeRunOutputs,
  parseWorkflowIdentity,
  requireWorkflowOutputs,
} from "../runs/output-model";
import { workflowFormats } from "../workflows/formats";
import { auditProfileLabel } from "./audits/labels";
import { ProjectRunIdentity } from "./run-history";

function RecentRunResults({
  summary,
  projectId,
}: {
  summary: RunSummary;
  projectId: string;
}) {
  const api = usePublicAPI();
  const identity = parseWorkflowIdentity(summary.workflow);
  const run = useQuery({
    queryKey: queryKeys.runs.detail(summary.runId),
    queryFn: () => getRun(api, summary.runId),
  });
  const contract = useQuery({
    queryKey: [
      ...queryKeys.workflows.detail(
        identity?.name ?? "",
        identity?.version ?? "",
      ),
      "outputs",
    ],
    queryFn: async () => {
      if (!identity) throw new Error("Workflow identity is invalid");
      return requireWorkflowOutputs(
        await getWorkflow(api, identity.name, identity.version),
        identity,
      );
    },
    enabled: identity !== undefined,
  });
  const entries = organizeRunOutputs(run.data?.outputs ?? {}, contract.data);
  const primary = entries.filter((entry) => entry.kind === "primary");
  const shown = (primary.length ? primary : entries).slice(0, 2);
  return (
    <div className="project-recent-result">
      <ContextLink
        returnLabel="Project Overview"
        to={`/runs/${encodeURIComponent(summary.runId)}`}
        className="project-result-workflow"
      >
        {summary.workflow} ↗
      </ContextLink>
      <small className="project-summary-caption">
        <code title={summary.runId}>
          {summary.runId.slice(0, 8)}…{summary.runId.slice(-8)}
        </code>
        {" · "}
        {formatTimestamp(summary.updatedAt)}
      </small>
      {run.isPending ? (
        <p className="loading-copy">Loading exact outputs…</p>
      ) : run.error ? (
        <>
          <ErrorNotice error={run.error} />
          <button
            type="button"
            className="secondary-button"
            onClick={() => void run.refetch()}
          >
            Retry outputs
          </button>
        </>
      ) : run.data.projectId !== projectId ? (
        <p className="form-error">Run does not belong to this project.</p>
      ) : (
        <>
          {contract.error ? (
            <p className="muted-copy">
              Output roles unavailable.{" "}
              <button
                type="button"
                className="secondary-button"
                onClick={() => void contract.refetch()}
              >
                Retry output roles
              </button>
            </p>
          ) : null}
          {shown.length === 0 ? (
            <p className="compact-empty">No output published by this Run.</p>
          ) : (
            shown.map((entry) => (
              <div className="project-overview-file" key={entry.slot}>
                {entry.artifact ? (
                  <ContextLink
                    returnLabel="Project Overview"
                    to={`/runs/${encodeURIComponent(summary.runId)}/artifacts/${encodeURIComponent(entry.artifact.namespace)}/${encodeURIComponent(entry.artifact.name)}?revision=${encodeURIComponent(entry.artifact.revision)}`}
                  >
                    <strong>{entry.slot}</strong>
                    <small>
                      {entry.kind === "primary"
                        ? "Primary result"
                        : entry.kind === "declared"
                          ? "Supporting result"
                          : "Run output"}
                      {entry.declaration
                        ? ` · ${entry.declaration.mediaTypes.map((type) => workflowFormats[type] ?? type).join(" / ")}`
                        : ""}
                    </small>
                  </ContextLink>
                ) : (
                  <p className="muted-copy">
                    {entry.kind === "primary" ? "Primary result" : "Output"}{" "}
                    <code>{entry.slot}</code> was not published.
                  </p>
                )}
              </div>
            ))
          )}
        </>
      )}
    </div>
  );
}

export function ProjectOverview({ project }: { project: Project }) {
  const api = usePublicAPI();
  const projectId = project.projectId;
  const root = `/projects/${encodeURIComponent(projectId)}`;
  const recentOptions = { limit: 5 };
  const resultOptions = { limit: 3, state: "succeeded" as const };
  const recent = useQuery({
    queryKey: queryKeys.projects.runView(projectId, recentOptions),
    queryFn: () => listProjectRuns(api, { projectId, ...recentOptions }),
    refetchInterval: 10_000,
  });
  const successful = useQuery({
    queryKey: queryKeys.projects.runView(projectId, resultOptions),
    queryFn: () => listProjectRuns(api, { projectId, ...resultOptions }),
    refetchInterval: 15_000,
  });
  const audits = useQuery({
    queryKey: [...queryKeys.projects.audits.all(projectId), "overview"],
    queryFn: () => listProjectAudits(api, { projectId, limit: 3 }),
    refetchInterval: 10_000,
  });
  const decisions = useQuery({
    queryKey: [...queryKeys.projects.audits.all(projectId), "waiting-review"],
    queryFn: () =>
      listProjectAudits(api, { projectId, state: "waiting_review", limit: 3 }),
    refetchInterval: 10_000,
  });
  const materials = useQuery({
    queryKey: [...queryKeys.projects.artifacts.all(projectId), "overview"],
    queryFn: () => listProjectArtifacts(api, { projectId, limit: 3 }),
  });
  const empty =
    materials.isSuccess &&
    materials.data.items.length === 0 &&
    !materials.data.page.hasMore;
  const queries = [recent, successful, audits, decisions, materials];
  return (
    <section className="project-overview-section">
      <header className="section-heading">
        <div>
          <h2>Overview</h2>
          <p className="muted-copy">
            Continue work, review results, or prepare the next analysis.
          </p>
        </div>
        <button
          type="button"
          className="secondary-button"
          disabled={queries.some((query) => query.isFetching)}
          onClick={() =>
            void Promise.all(queries.map((query) => query.refetch()))
          }
        >
          Refresh overview
        </button>
      </header>
      <div className="project-overview-grid">
        <div className="project-overview-stack">
          <section className="panel project-overview-attention">
            <p className="eyebrow">Needs your attention</p>
            {decisions.isPending ? (
              <p className="loading-copy">Checking Audit decisions…</p>
            ) : decisions.error ? (
              <ErrorNotice error={decisions.error} />
            ) : decisions.data.items.length ? (
              <>
                <h3>Audits waiting for review</h3>
                {decisions.data.items.map((audit) => (
                  <div className="project-overview-row" key={audit.auditId}>
                    <strong>{auditProfileLabel(audit)}</strong>
                    <ContextLink
                      returnLabel="Project Overview"
                      className="audit-open-link"
                      to={`${root}/audits/${encodeURIComponent(audit.auditId)}/reviews?state=pending`}
                    >
                      Review decisions →
                    </ContextLink>
                  </div>
                ))}
                {decisions.data.page.hasMore ? (
                  <Link to={`${root}/audits`}>All audits →</Link>
                ) : null}
              </>
            ) : empty ? (
              <>
                <h3>Prepare your first analysis</h3>
                <p>
                  Add source material, choose a Workflow and review its inputs
                  before starting.
                </p>
                <Link
                  className="audit-open-link"
                  to={`${root}/artifacts?add=artifact`}
                >
                  Add sources →
                </Link>
              </>
            ) : (
              <>
                <h3>No Audits are waiting for review</h3>
                <p>
                  Choose a Workflow to continue analysis, or inspect your Audit
                  progress.
                </p>
                <Link className="audit-open-link" to={`${root}/workflows`}>
                  Choose workflow →
                </Link>
              </>
            )}
          </section>
          <section className="panel">
            <div className="project-overview-heading">
              <h3>Recent results</h3>
              <Link to={`${root}/runs?view=completed&state=succeeded`}>
                Successful Runs →
              </Link>
            </div>
            <p className="project-summary-caption">
              Outputs from the three most recent successful Runs.
            </p>
            {successful.isPending ? (
              <p className="loading-copy">Loading successful Runs…</p>
            ) : successful.error ? (
              <ErrorNotice error={successful.error} />
            ) : successful.data.items.length === 0 ? (
              <p className="compact-empty">
                No successful Runs yet. Published results will appear here.
              </p>
            ) : (
              successful.data.items.map((run) => (
                <RecentRunResults
                  key={run.runId}
                  summary={run}
                  projectId={projectId}
                />
              ))
            )}
          </section>
          <section className="panel">
            <div className="project-overview-heading">
              <h3>Recent Runs</h3>
              <Link to={`${root}/runs`}>All Runs →</Link>
            </div>
            {recent.isPending ? (
              <p className="loading-copy">Loading recent Runs…</p>
            ) : recent.error ? (
              <ErrorNotice error={recent.error} />
            ) : recent.data.items.length === 0 ? (
              <p className="compact-empty">
                No Runs yet. Choose a Workflow when your inputs are ready.
              </p>
            ) : (
              recent.data.items.map((run) => (
                <div className="project-overview-row" key={run.runId}>
                  <div>
                    <ProjectRunIdentity
                      run={run}
                      returnLabel="Project Overview"
                    />
                    <small>{formatTimestamp(run.updatedAt)}</small>
                  </div>
                  <StateBadge state={run.state} />
                </div>
              ))
            )}
          </section>
        </div>
        <div className="project-overview-stack">
          <section className="panel">
            <div className="project-overview-heading">
              <h3>Project context</h3>
              <Link to={`${root}/settings`}>Settings →</Link>
            </div>
            <p className="eyebrow">Materials</p>
            {materials.isPending ? (
              <p className="loading-copy">Loading materials…</p>
            ) : materials.error ? (
              <ErrorNotice error={materials.error} />
            ) : empty ? (
              <p className="compact-empty">No materials added yet.</p>
            ) : (
              <>
                {materials.data.items.map((item) => (
                  <div
                    className="project-overview-file"
                    key={`${item.artifact.namespace}/${item.artifact.name}`}
                  >
                    <ContextLink
                      returnLabel="Project Overview"
                      to={`${root}/artifacts/${encodeURIComponent(item.artifact.namespace)}/${encodeURIComponent(item.artifact.name)}?revision=${encodeURIComponent(item.artifact.revision)}`}
                    >
                      <strong>
                        {item.artifact.namespace}/{item.artifact.name}
                      </strong>
                      <small>
                        {workflowFormats[item.mediaType] ?? item.mediaType}
                      </small>
                    </ContextLink>
                  </div>
                ))}
                <Link className="project-summary-link" to={`${root}/artifacts`}>
                  All artifacts →
                </Link>
              </>
            )}
            <dl className="project-overview-target">
              <dt>Application target</dt>
              <dd>{project.httpTarget?.url ?? "Not configured"}</dd>
            </dl>
            <div className="project-overview-actions">
              <Link to={`${root}/artifacts?add=artifact`}>Add sources →</Link>
              <Link to={`${root}/workflows`}>Choose workflow →</Link>
            </div>
          </section>
          <section className="panel">
            <div className="project-overview-heading">
              <h3>Recent audits</h3>
              <Link to={`${root}/audits`}>All audits →</Link>
            </div>
            {audits.isPending ? (
              <p className="loading-copy">Loading Audits…</p>
            ) : audits.error ? (
              <ErrorNotice error={audits.error} />
            ) : audits.data.items.length === 0 ? (
              <p className="compact-empty">No Audits yet.</p>
            ) : (
              audits.data.items.map((audit) => (
                <div className="project-overview-row" key={audit.auditId}>
                  <div>
                    <ContextLink
                      returnLabel="Project Overview"
                      to={`${root}/audits/${encodeURIComponent(audit.auditId)}`}
                    >
                      {auditProfileLabel(audit)}
                    </ContextLink>
                    {audit.stopReason ? (
                      <small>
                        {audit.stopReason.code === "deadline_exhausted"
                          ? "Time limit reached"
                          : audit.stopReason.message}
                      </small>
                    ) : null}
                  </div>
                  <StateBadge state={audit.state} />
                </div>
              ))
            )}
          </section>
        </div>
      </div>
    </section>
  );
}
