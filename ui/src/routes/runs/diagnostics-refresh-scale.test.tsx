import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { queryKeys } from "../../api/query-keys";
import { useRuntimeDiagnosticRefresh } from "./runtime-diagnostics-refresh";

describe("diagnostic entry work on complete retained histories", () => {
  it.each([false, true])(
    "bounds cache scans for 1000 attempts, cached facts=%s",
    (warm) => {
      const client = new QueryClient({
        defaultOptions: { queries: { gcTime: Infinity } },
      });
      const cache = client.getQueryCache();
      const attempts = Array.from({ length: 1000 }, (_, index) => ({
        stageExecutionId: `stage-${index}`,
        state: "succeeded" as const,
      }));
      const data = { pages: [], pageParams: [] };
      for (const attempt of attempts) {
        // Both disabled widgets and local selection exist even when closed.
        cache.build(client, {
          queryKey: queryKeys.runs.runtimeDiagnostics(
            "run-1",
            attempt.stageExecutionId,
          ),
        });
        cache.build(client, {
          queryKey: [
            ...queryKeys.runs.toolCalls("run-1", attempt.stageExecutionId),
            "planner",
            null,
          ],
        });
        client.setQueryData(
          queryKeys.runs.toolCallSelection("run-1", attempt.stageExecutionId),
          "planner",
        );
      }
      const runtime = queryKeys.runs.runtimeDiagnostics("run-1", "stage-500");
      const tools = [
        ...queryKeys.runs.toolCalls("run-1", "stage-500"),
        "planner",
        null,
      ];
      const other = queryKeys.runs.runtimeDiagnostics("other-run", "stage-500");
      client.setQueryData(other, data);
      if (warm) {
        client.setQueryData(runtime, data);
        client.setQueryData(tools, data);
      }
      const invalidations = vi.spyOn(client, "invalidateQueries");
      const scans = vi.spyOn(cache, "findAll");
      function Body() {
        useRuntimeDiagnosticRefresh("run-1", attempts);
        return null;
      }
      const rendered = render(
        <QueryClientProvider client={client}>
          <Body />
        </QueryClientProvider>,
      );
      try {
        expect(invalidations).toHaveBeenCalledTimes(warm ? 1 : 0);
        expect(scans.mock.calls.length).toBeLessThanOrEqual(3);
        expect(client.getQueryState(runtime)?.isInvalidated).toBe(warm);
        expect(client.getQueryState(tools)?.isInvalidated).toBe(warm);
        expect(client.getQueryState(other)?.isInvalidated).toBe(false);
        expect(
          client.getQueryState(
            queryKeys.runs.runtimeDiagnostics("run-1", "stage-999"),
          )?.isInvalidated,
        ).toBe(false);
        expect(
          client.getQueryState(
            queryKeys.runs.toolCallSelection("run-1", "stage-500"),
          )?.isInvalidated,
        ).toBe(false);
      } finally {
        rendered.unmount();
        client.clear();
      }
    },
  );
});
