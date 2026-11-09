import type { CostScope } from "../api/llm-cost";
import { QueryView } from "./query-view";
import { useLLMCost, type CostQueryOptions } from "./llm-cost-query";
import { LLMCostCard } from "./llm-cost-card";

export function LLMCostPanel({
  scope,
  id,
  ...options
}: { scope: CostScope; id: string } & CostQueryOptions) {
  const query = useLLMCost(scope, id, options);
  return (
    <div className="llm-cost-panel">
      <QueryView
        query={query}
        errorContext="Could not load LLM cost"
        loading={
          <section className="llm-cost-card" aria-label="LLM cost">
            <div className="llm-cost-shell">
              <h2>LLM cost</h2>
              <p className="ui-caption" role="status">
                Loading cost…
              </p>
            </div>
          </section>
        }
      >
        {(value) => (
          <LLMCostCard
            key={`${scope}:${id}`}
            value={value}
            refreshing={query.isFetching}
            onRefresh={() => void query.refetch()}
          />
        )}
      </QueryView>
    </div>
  );
}
