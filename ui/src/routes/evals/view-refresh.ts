import { useQueryClient } from "@tanstack/react-query";
import { useLayoutEffect, useRef } from "react";
import { useLocation, useSearchParams } from "react-router";

import type { EvalExperiment } from "../../api/evals";
import { queryKeys } from "../../api/query-keys";

export function useEvalViewRefresh(
  experiment: EvalExperiment,
  view: "comparison" | "attempts",
) {
  const cache = useQueryClient();
  const { pathname } = useLocation();
  const [params, setParams] = useSearchParams();
  const current = useRef<{
    params: URLSearchParams;
    setParams: typeof setParams;
  } | null>(null);
  const generation = useRef(0);

  // A retained React Router setter (including its updater) captures old params.
  // Only committed renders may supply the URL to patch after an awaited read.
  useLayoutEffect(() => {
    current.current = { params, setParams };
  });
  useLayoutEffect(
    () => () => {
      current.current = null;
      generation.current++;
    },
    [experiment.experimentId, pathname, view],
  );

  return async function refresh() {
    const request = ++generation.current;
    const experimentKey = queryKeys.evals.experiment(experiment.experimentId);
    await cache.invalidateQueries({ queryKey: experimentKey });
    const context = current.current;
    if (context === null || request !== generation.current) return;

    const next = new URLSearchParams(context.params);
    next.delete("cursor");
    if (view === "comparison") {
      const latest =
        cache.getQueryData<EvalExperiment>(experimentKey) ?? experiment;
      if (latest.viewSnapshot) next.set("viewSnapshot", latest.viewSnapshot);
      else next.delete("viewSnapshot");
      next.delete("chartCursor");
      next.delete("binFilter");
    } else {
      next.delete("viewSnapshot");
    }
    context.setParams(next);
    await Promise.all(
      (view === "comparison"
        ? (["pairs", "chart"] as const)
        : (["members"] as const)
      ).map((projection) =>
        cache.invalidateQueries({
          queryKey: queryKeys.evals.projection(
            projection,
            experiment.experimentId,
          ),
        }),
      ),
    );
  };
}
