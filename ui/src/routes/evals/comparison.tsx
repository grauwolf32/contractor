import { useMutation, useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { useSearchParams } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  getEvalReport,
  listEvalPairs,
  type EvalExperiment,
  type EvalPairQuery,
} from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { ContextLink } from "../../app/context-navigation";
import { saveBlob } from "../../app/download";
import { RefreshButton } from "../../app/refresh-button";
import { Button, Field, FilterChips, StatusChip } from "../../ui";
import { EvalChartPanel } from "./charts";
import { ArmKey, EvalError } from "./common";
import { PAIR_FILTERS } from "./labels";
import { MemberSummary } from "./member";
import { useEvalViewRefresh } from "./view-refresh";

export function EvalComparison({ experiment }: { experiment: EvalExperiment }) {
  const api = usePublicAPI();
  const chartsHeading = useId(),
    pairsHeading = useId();
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
    queryKey: queryKeys.evals.pairs(experiment.experimentId, query),
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
      saveBlob(
        new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
        `eval-${experiment.experimentId}.json`,
      );
    },
  });
  if (!snapshot)
    return (
      <p className="ui-caption">
        Comparison is waiting for a prepared plan and collected observations.
      </p>
    );
  return (
    <>
      <section className="eval-section" aria-labelledby={chartsHeading}>
        <div className="eval-section-heading">
          <h2 id={chartsHeading}>Charts</h2>
          <div className="ui-toolbar">
            <RefreshButton
              className="ui-btn"
              isFetching={false}
              onRefresh={() => void refresh()}
              label="Refresh"
            />
            <Button pending={report.isPending} onClick={() => report.mutate()}>
              Export safe report
            </Button>
          </div>
        </div>
        <EvalError error={report.error} />
        <div className="eval-inline-fields">
          <Field label="Comparison metric">
            {(control) => (
              <select
                {...control}
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
            )}
          </Field>
          <Field label="Chart view">
            {(control) => (
              <select
                {...control}
                value={plot === "pair-deltas" ? "pair-deltas" : "distribution"}
                onChange={(e) =>
                  update({ plot: e.target.value, chartCursor: null })
                }
              >
                <option value="distribution">Distribution</option>
                <option value="pair-deltas">Case differences</option>
              </select>
            )}
          </Field>
        </div>
        <p className="ui-caption">
          Charts describe the complete matching cohort in this snapshot.
          Pair-list filters below keep whole-experiment denominators unchanged.
        </p>
        <EvalChartPanel
          experiment={experiment}
          chart={plot}
          metric={metric}
          snapshot={snapshot}
          onRefresh={() => void refresh()}
          cursor={chartCursor ?? undefined}
          onNext={(next) => update({ chartCursor: next })}
          onBin={(token) =>
            update({ binFilter: token, cursor: null, filter: "all" })
          }
        />
        {chartCursor ? (
          <div className="ui-toolbar">
            <Button size="sm" onClick={() => update({ chartCursor: null })}>
              First differences
            </Button>
          </div>
        ) : null}
      </section>
      <section className="eval-section" aria-labelledby={pairsHeading}>
        <h2 id={pairsHeading}>Paired evidence</h2>
        <FilterChips
          label="Pair filter"
          options={PAIR_FILTERS}
          value={filter}
          onChange={(value) => update({ filter: value, cursor: null })}
        />
        {binFilter ? (
          <p className="eval-callout">
            Filtered to the selected distribution bin.{" "}
            <Button
              size="xs"
              onClick={() => update({ binFilter: null, cursor: null })}
            >
              Clear bin filter
            </Button>
          </p>
        ) : null}
        <EvalError error={pairs.error} reload={() => void refresh()} />
        {pairs.data ? (
          <>
            <p className="ui-caption">
              {pairs.data.filteredCount} matching pairs · {pairs.data.freshness}
              . Complete quality pairs:{" "}
              {pairs.data.experimentSummary.completeQualityPairs}; complete
              token pairs: {pairs.data.experimentSummary.completeTokenPairs}.
            </p>
            {pairs.data.items.length ? (
              <ul className="eval-records" role="list">
                {pairs.data.items.map((pair) => (
                  <li className="eval-record eval-pair" key={pair.pairId}>
                    <h3 className="eval-record-title">
                      <ContextLink
                        returnLabel="Comparison"
                        to={`/evals/experiments/${encodeURIComponent(experiment.experimentId)}/pairs/${encodeURIComponent(pair.pairId)}?viewSnapshot=${encodeURIComponent(pairs.data!.viewSnapshot)}`}
                      >
                        {pair.caseId} / sample {pair.sample}
                      </ContextLink>
                      {pair.regression ? (
                        <StatusChip tone="blocked" size="sm">
                          Quality regression
                        </StatusChip>
                      ) : null}
                    </h3>
                    <div className="eval-arms">
                      <section className="eval-arm-column" data-arm="a">
                        <h4>
                          <ArmKey arm="a">A · Baseline</ArmKey>
                        </h4>
                        <MemberSummary member={pair.a} />
                      </section>
                      <section className="eval-arm-column" data-arm="b">
                        <h4>
                          <ArmKey arm="b">B · Candidate</ArmKey>
                        </h4>
                        <MemberSummary member={pair.b} />
                      </section>
                    </div>
                    {pair.exclusions.map((reason) => (
                      <p className="ui-caption" key={reason}>
                        {reason.replaceAll("_", " ")}
                      </p>
                    ))}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="ui-caption">
                No pairs match this filter. Choose All pairs or Unresolved pairs
                to inspect the remaining evidence.
              </p>
            )}
            <div className="ui-toolbar">
              <Button
                size="sm"
                disabled={!cursor}
                onClick={() => update({ cursor: null })}
              >
                First pairs
              </Button>
              <Button
                size="sm"
                disabled={!pairs.data.page.hasMore}
                onClick={() => update({ cursor: pairs.data!.page.nextCursor })}
              >
                Next pairs
              </Button>
            </div>
          </>
        ) : null}
      </section>
    </>
  );
}
