import { useMutation, useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  getEvalReport,
  listEvalPairs,
  type EvalExperiment,
  type EvalPairQuery,
} from "../../api/evals";
import { ContextLink } from "../../app/context-navigation";
import { EvalChartPanel } from "./charts";
import { EvalError, EvalField } from "./common";
import { MemberSummary } from "./member";
import { useEvalViewRefresh } from "./view-refresh";

export function EvalComparison({ experiment }: { experiment: EvalExperiment }) {
  const api = usePublicAPI();
  const [params, setParams] = useSearchParams();
  const refresh = useEvalViewRefresh(experiment, "comparison");
  const snapshot = params.get("viewSnapshot") ?? experiment.viewSnapshot;
  const filter =
    params.get("filter") === "all"
      ? "all"
      : params.get("filter") === "unresolved"
        ? "unresolved"
        : "regressions";
  const metric = params.get("metric") === "duration" ? "duration" : "tokens";
  const plot = params.get("plot") === "pair-deltas" ? "pair-deltas" : metric;
  const cursor = params.get("cursor"),
    binFilter = params.get("binFilter"),
    chartCursor = params.get("chartCursor");
  const query: EvalPairQuery = {
    filter,
    ...(snapshot ? { viewSnapshot: snapshot } : {}),
    ...(cursor ? { cursor } : {}),
    ...(binFilter ? { binFilter } : {}),
  };
  const pairs = useQuery({
    queryKey: ["evals", "pairs", experiment.experimentId, query],
    enabled: !!snapshot,
    queryFn: ({ signal }) =>
      listEvalPairs(api, experiment.experimentId, query, signal),
  });
  function update(patch: Record<string, string | null>) {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(patch)) {
      if (value) next.set(key, value);
      else next.delete(key);
    }
    if (snapshot && !Object.hasOwn(patch, "viewSnapshot"))
      next.set("viewSnapshot", snapshot);
    setParams(next);
  }
  const report = useMutation({
    mutationFn: async () => {
      const data = await getEvalReport(api, experiment.experimentId, snapshot!);
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = `eval-${experiment.experimentId}.json`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    },
  });
  if (!snapshot)
    return (
      <p>
        Comparison is waiting for a prepared plan and collected observations.
      </p>
    );
  return (
    <>
      <div className="eval-actions">
        <button
          type="button"
          className="secondary-button"
          onClick={() => void refresh()}
        >
          Refresh comparison snapshot
        </button>
        <button
          type="button"
          className="secondary-button"
          disabled={report.isPending}
          onClick={() => report.mutate()}
        >
          Export safe report
        </button>
      </div>
      <EvalError error={report.error} />
      <div className="form-grid">
        <EvalField label="Comparison metric">
          <select
            value={metric}
            onChange={(e) =>
              update({
                metric: e.target.value,
                chartCursor: null,
                binFilter: null,
                cursor: null,
              })
            }
          >
            <option value="tokens">Tokens</option>
            <option value="duration">Duration</option>
          </select>
        </EvalField>
        <EvalField label="Chart view">
          <select
            value={plot === "pair-deltas" ? "pair-deltas" : "distribution"}
            onChange={(e) =>
              update({ plot: e.target.value, chartCursor: null })
            }
          >
            <option value="distribution">Distribution</option>
            <option value="pair-deltas">Case differences</option>
          </select>
        </EvalField>
      </div>
      <p>
        Charts describe the complete matching cohort in this snapshot. Pair-list
        filters below keep whole-experiment denominators unchanged.
      </p>
      <EvalChartPanel
        experiment={experiment}
        chart={plot}
        metric={metric}
        snapshot={snapshot}
        cursor={chartCursor ?? undefined}
        onNext={(next) => update({ chartCursor: next })}
        onBin={(token) =>
          update({ binFilter: token, cursor: null, filter: "all" })
        }
      />
      {chartCursor ? (
        <button
          type="button"
          className="secondary-button"
          onClick={() => update({ chartCursor: null })}
        >
          First differences
        </button>
      ) : null}
      <h2>Paired evidence</h2>
      <EvalField label="Pair filter">
        <select
          value={filter}
          onChange={(e) => update({ filter: e.target.value, cursor: null })}
        >
          <option value="regressions">Quality regressions</option>
          <option value="unresolved">Unresolved pairs</option>
          <option value="all">All pairs</option>
        </select>
      </EvalField>
      {binFilter ? (
        <p>
          Filtered to the selected distribution bin.{" "}
          <button
            type="button"
            className="secondary-button"
            onClick={() => update({ binFilter: null, cursor: null })}
          >
            Clear bin filter
          </button>
        </p>
      ) : null}
      <EvalError error={pairs.error} reload={() => void refresh()} />
      {pairs.data ? (
        <>
          <p>
            {pairs.data.filteredCount} matching pairs · {pairs.data.freshness}.
            Complete quality pairs:{" "}
            {pairs.data.experimentSummary.completeQualityPairs}; complete token
            pairs: {pairs.data.experimentSummary.completeTokenPairs}.
          </p>
          <div className="eval-pairs">
            {pairs.data.items.map((pair) => (
              <article className="panel eval-panel" key={pair.pairId}>
                <h3>
                  <ContextLink
                    returnLabel="Comparison"
                    to={`/evals/experiments/${encodeURIComponent(experiment.experimentId)}/pairs/${encodeURIComponent(pair.pairId)}?viewSnapshot=${encodeURIComponent(pairs.data!.viewSnapshot)}`}
                  >
                    {pair.caseId} / sample {pair.sample}
                  </ContextLink>
                  {pair.regression ? " · Quality regression" : ""}
                </h3>
                <div className="eval-variants">
                  <section className="eval-arm-a">
                    <h4>A · Baseline</h4>
                    <MemberSummary member={pair.a} />
                  </section>
                  <section className="eval-arm-b">
                    <h4>B · Candidate</h4>
                    <MemberSummary member={pair.b} />
                  </section>
                </div>
                {pair.exclusions.map((reason) => (
                  <p key={reason}>{reason.replaceAll("_", " ")}</p>
                ))}
              </article>
            ))}
          </div>
          {!pairs.data.items.length ? (
            <p>
              No pairs match this filter. Choose All pairs or Unresolved pairs
              to inspect the remaining evidence.
            </p>
          ) : null}
          <div className="eval-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={!cursor}
              onClick={() => update({ cursor: null })}
            >
              First pairs
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={!pairs.data.page.hasMore}
              onClick={() => update({ cursor: pairs.data!.page.nextCursor })}
            >
              Next pairs
            </button>
          </div>
        </>
      ) : null}
    </>
  );
}
