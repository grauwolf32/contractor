import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";

import { usePublicAPI } from "../../../api/context";
import { listAllocationResourceHistory } from "../../../api/performance";
import { queryKeys } from "../../../api/query-keys";
import { RUN_ID_PATTERN } from "../../../api/runs";
import { CursorControls, ErrorNotice } from "../../artifacts/common";
import { AllocationResourceList } from "../performance/resources";
import { AllocationViewTabs } from "./tabs";

export function CompletedAllocationListRoute() {
  const api = usePublicAPI();
  const [draftRunId, setDraftRunId] = useState("");
  const [runId, setRunId] = useState<string>();
  const [validationError, setValidationError] = useState<string>();
  const [cursors, setCursors] = useState<string[]>([]);
  const cursor = cursors.at(-1);
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
      setValidationError("Run ID must be an exact public resource identity.");
      return;
    }
    setRunId(value === "" ? undefined : value);
    setCursors([]);
    setValidationError(undefined);
  }

  return (
    <div className="operations-library">
      <AllocationViewTabs />
      <section className="panel completed-allocation-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Terminal StageExecution history</p>
            <h3>Completed allocation resources</h3>
            <p className="muted-copy">
              Entries survive authoritative release and retain the collection
              policy selected for that allocation. This view has no lifecycle
              mutation controls.
            </p>
          </div>
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh history"}
          </button>
        </div>
        <form className="allocation-history-filter" onSubmit={applyFilter}>
          <label>
            <span>Exact Run ID (optional)</span>
            <input
              value={draftRunId}
              aria-invalid={validationError !== undefined}
              placeholder="run_…"
              onChange={(event) => {
                setDraftRunId(event.target.value);
                setValidationError(undefined);
              }}
            />
          </label>
          <button className="secondary-button" type="submit">
            Apply filter
          </button>
          {runId === undefined ? null : (
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                setDraftRunId("");
                setRunId(undefined);
                setCursors([]);
                setValidationError(undefined);
              }}
            >
              Clear
            </button>
          )}
        </form>
        {validationError === undefined ? null : (
          <p className="field-error" role="alert">
            {validationError}
          </p>
        )}
        {query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading completed allocations…
          </p>
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No matching terminal allocation exists.</strong>
            <p>
              Missing reports are included, so an empty result means no owned
              terminal allocation matches this page and filter.
            </p>
          </div>
        ) : (
          <AllocationResourceList items={query.data.items} />
        )}
        {query.data === undefined ? null : (
          <CursorControls
            label="Completed allocation pages"
            canGoBack={cursors.length > 0}
            {...(query.data.page.nextCursor === undefined
              ? {}
              : { nextCursor: query.data.page.nextCursor })}
            onBack={() =>
              setCursors((current) =>
                current.slice(0, Math.max(0, current.length - 1)),
              )
            }
            onNext={(next) => setCursors((current) => [...current, next])}
          />
        )}
      </section>
    </div>
  );
}
