import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  EVAL_POLL_MS,
  listEvalExperiments,
  type EvalListQuery,
} from "../../api/evals";
import { EvalError, EvalField, EvalFrame } from "./common";
import { useEvalProjects } from "./queries";

const EVAL_STATES = [
  "draft",
  "preparing",
  "ready",
  "running",
  "settling",
  "finished",
  "pausing",
  "paused",
  "cancelling",
  "cancelled",
  "interrupted",
] as const;

export function EvalListRoute() {
  const api = usePublicAPI(),
    projects = useEvalProjects();
  const [params, setParams] = useSearchParams();
  const state = params.get("state"),
    mode = params.get("controlMode"),
    cursor = params.get("cursor"),
    projectId = params.get("projectId"),
    datasetId = params.get("datasetId");
  const query: EvalListQuery = {
    ...(EVAL_STATES.some((x) => x === state)
      ? { state: state as NonNullable<EvalListQuery["state"]> }
      : {}),
    ...(mode === "server" || mode === "external" ? { controlMode: mode } : {}),
    ...(cursor ? { cursor } : {}),
    ...(projectId ? { projectId } : {}),
    ...(datasetId ? { datasetId } : {}),
  };
  const list = useQuery({
    queryKey: ["evals", "list", query],
    queryFn: ({ signal }) => listEvalExperiments(api, query, signal),
    refetchInterval: cursor ? false : EVAL_POLL_MS,
  });
  function filter(key: string, value: string) {
    const next = new URLSearchParams(params);
    next.delete("cursor");
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  }
  return (
    <EvalFrame
      title="Experiments"
      action={
        <Link className="button-link" to="/evals/new">
          New experiment
        </Link>
      }
    >
      <nav className="eval-actions" aria-label="Evaluation collections">
        <Link to="/evals/datasets">Datasets</Link>
        <Link to="/evals/legacy">Legacy evaluation workspaces</Link>
      </nav>
      <div className="form-grid">
        <EvalField label="Lifecycle">
          <select
            value={state ?? ""}
            onChange={(e) => filter("state", e.target.value)}
          >
            <option value="">All states</option>
            {EVAL_STATES.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </EvalField>
        <EvalField label="Control mode">
          <select
            value={mode ?? ""}
            onChange={(e) => filter("controlMode", e.target.value)}
          >
            <option value="">All producers</option>
            <option value="server">Native server</option>
            <option value="external">External producer</option>
          </select>
        </EvalField>
        <EvalField label="Evaluation workspace">
          <select
            value={projectId ?? ""}
            onChange={(e) => filter("projectId", e.target.value)}
          >
            <option value="">All workspaces</option>
            {projects.data?.map((p) => (
              <option key={p.projectId} value={p.projectId}>
                {p.name}
              </option>
            ))}
          </select>
        </EvalField>
        <EvalField label="Dataset ID">
          <input
            value={datasetId ?? ""}
            onChange={(e) => filter("datasetId", e.target.value)}
          />
        </EvalField>
      </div>
      <EvalError
        error={list.error ?? projects.error}
        reload={() => {
          filter("cursor", "");
          void list.refetch();
        }}
      />
      {list.isPending ? <p role="status">Loading experiments…</p> : null}
      {list.data?.items.length === 0 ? (
        <section className="panel eval-panel">
          <h2>No matching experiments</h2>
          <p>
            Create an experiment or change the filters. Earlier evaluation
            workspaces remain in legacy history.
          </p>
        </section>
      ) : null}
      <div className="eval-experiment-list">
        {list.data?.items.map((item) => (
          <article className="panel eval-panel" key={item.experimentId}>
            <div className="section-heading">
              <h2>
                <Link
                  to={`/evals/experiments/${encodeURIComponent(item.experimentId)}/${item.state === "draft" ? "setup" : "overview"}`}
                >
                  {item.name}
                </Link>
              </h2>
              <span>
                {item.state} · {item.controlMode}
              </span>
            </div>
            <p>
              {item.executionKind} · {item.expectedMembers} expected members
              {item.caseCount && item.repetitions
                ? ` · ${item.caseCount} cases × 2 × ${item.repetitions}`
                : ""}
            </p>
            {item.variants?.map((v, i) => (
              <p key={v.id}>
                {i === 0 ? "A" : "B"}: {v.selector}
              </p>
            ))}
            {item.summary ? (
              <>
                <p>
                  Conclusion: {item.summary.conclusion} · {item.freshness}
                </p>
                {Object.entries(item.summary.counts).map(([arm, counts]) => (
                  <p key={arm}>
                    {arm}: {counts.terminal}/{counts.expected} terminal ·{" "}
                    {counts.scored}/{counts.expected} scored ·{" "}
                    {counts.endToEndPassed}/{counts.expected} passed
                  </p>
                ))}
              </>
            ) : (
              <p>Comparison pending</p>
            )}
            <small>Updated {new Date(item.updatedAt).toLocaleString()}</small>
          </article>
        ))}
      </div>
      <nav className="eval-actions" aria-label="Experiment pages">
        <button
          type="button"
          className="secondary"
          disabled={!cursor}
          onClick={() => filter("cursor", "")}
        >
          First page
        </button>
        <button
          type="button"
          className="secondary"
          disabled={!list.data?.page.hasMore}
          onClick={() => {
            if (list.data?.page.nextCursor) {
              const next = new URLSearchParams(params);
              next.set("cursor", list.data.page.nextCursor);
              setParams(next);
            }
          }}
        >
          Next page
        </button>
      </nav>
    </EvalFrame>
  );
}
