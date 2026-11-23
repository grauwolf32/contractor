import { hashKey, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { usePublicAPI } from "../api/context";
import { getLLMCost, type CostScope, type LLMCost } from "../api/llm-cost";
import { pollInterval } from "./polling";
import { queryKeys } from "../api/query-keys";

export interface CostQueryOptions {
  active?: boolean | undefined;
  refreshKey?: string | undefined;
  executionIds?: readonly string[] | undefined;
  enabled?: boolean | undefined;
}

export function useLLMCost(
  scope: CostScope,
  id: string,
  options: CostQueryOptions = {},
) {
  const api = usePublicAPI(),
    client = useQueryClient();
  const ids = [...(options.executionIds ?? [])].sort();
  const key = queryKeys.llmCost(scope, id, ids);
  const keyHash = hashKey(key);
  // Keys are rebuilt by callers; identity follows their contents.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const stableKey = useMemo(() => key, [keyHash]);
  const query = useQuery({
    queryKey: stableKey,
    enabled: options.enabled ?? true,
    queryFn: ({ signal }) => getLLMCost(api, scope, id, signal, ids),
    refetchInterval: pollInterval<LLMCost>({
      interval: 5000,
      settled: (value) => !value.pending && options.active !== true,
    }),
  });
  const before = useRef({ keyHash, refresh: options.refreshKey });
  useEffect(() => {
    const old = before.current;
    before.current = { keyHash, refresh: options.refreshKey };
    if (
      options.enabled === false ||
      options.active === true ||
      old.keyHash !== keyHash ||
      old.refresh === options.refreshKey
    )
      return;
    // Finish with a fresh independent cost read when the owning work settles.
    const exact = { queryKey: stableKey, exact: true };
    void client
      .cancelQueries(exact)
      .then(() => client.invalidateQueries(exact));
  }, [
    client,
    keyHash,
    stableKey,
    options.active,
    options.enabled,
    options.refreshKey,
  ]);
  return query;
}
