import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useId, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import { listAllocationResourceHistory } from "../../../api/performance";
import { queryKeys } from "../../../api/query-keys";
import { RUN_ID_PATTERN } from "../../../api/runs";
import { CursorControls } from "../../../app/cursor-controls";
import { useCursorStack } from "../../../app/pagination";
import { AllocationResourceList } from "../performance/resources";
import { AllocationViewTabs } from "./tabs";
import { RefreshButton } from "../../../app/refresh-button";
import { QueryView } from "../../../app/query-view";
import { OpsSection } from "../common";

export function CompletedAllocationListRoute() {
  const api = usePublicAPI();
  const errorId = useId();
  const [draftRunId, setDraftRunId] = useState("");
  const [runId, setRunId] = useState<string>();
  const [validationError, setValidationError] = useState<string>();
  const pages = useCursorStack();
  const cursor = pages.cursor;
  const query = useQuery({
    queryKey: queryKeys.operations.allocationHistory.list(runId, cursor),
    queryFn: ({ signal }) =>
      listAllocationResourceHistory(
        api,
        {
          ...(runId === undefined ? {} : { runId }),
          ...(cursor === undefined ? {} : { cursor }),
        },
        signal,
      ),
  });

  function applyFilter(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const value = draftRunId.trim();
    if (value !== "" && !RUN_ID_PATTERN.test(value)) {
      setValidationError("Run ID must be a valid Run identifier.");
      return;
    }
    setRunId(value === "" ? undefined : value);
    pages.reset();
    setValidationError(undefined);
  }

  return (
    <div className="ops-stack">
      <AllocationViewTabs />
      <OpsSection
        id="completed-allocations-heading"
        eyebrow="Allocation history"
        title="Completed allocation resources"
        description="Historical allocation measurements. Open metrics to inspect their scope, collection policy and identity."
        actions={
          <RefreshButton
            isFetching={query.isFetching}
            onRefresh={() => void query.refetch()}
            label="Refresh"
          />
        }
      >
        <form className="ops-filter-row" onSubmit={applyFilter}>
          <label className="ops-field">
            <span>Run ID (optional)</span>
            <input
              value={draftRunId}
              aria-invalid={validationError !== undefined}
              aria-describedby={
                validationError === undefined ? undefined : errorId
              }
              placeholder="run_…"
              onChange={(event) => {
                setDraftRunId(event.target.value);
                setValidationError(undefined);
              }}
            />
          </label>
          <button className="ui-btn" type="submit">
            Apply filter
          </button>
          {runId === undefined ? null : (
            <button
              className="ui-btn"
              data-variant="ghost"
              type="button"
              onClick={() => {
                setDraftRunId("");
                setRunId(undefined);
                pages.reset();
                setValidationError(undefined);
              }}
            >
              Clear
            </button>
          )}
        </form>
        {validationError === undefined ? null : (
          <p id={errorId} className="ops-field-error" role="alert">
            {validationError}
          </p>
        )}
        <QueryView
          query={query}
          loading={
            <p className="ops-loading" role="status">
              Loading completed allocations…
            </p>
          }
          onRetry={() => void query.refetch()}
          isEmpty={(data) => data.items.length === 0}
          empty={
            <div className="ops-empty">
              <strong>No matching terminal allocation exists.</strong>
              <p>
                Missing reports are included, so an empty result means no owned
                terminal allocation matches this page and filter.
              </p>
            </div>
          }
        >
          {(data) => <AllocationResourceList items={data.items} compact />}
        </QueryView>
        {query.data === undefined ? null : (
          <CursorControls
            label="Completed allocation pages"
            {...pages.controls(query.data.page)}
          />
        )}
      </OpsSection>
    </div>
  );
}
