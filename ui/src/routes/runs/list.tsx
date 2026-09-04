import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  normalizeRunMetadataLabelSelectors,
  RUN_METADATA_LABEL_LIMIT,
  type RunMetadataLabelSelector,
} from "../../api/run-metadata-labels";
import { listRuns, RUN_STATES, type WorkflowRunState } from "../../api/runs";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "./components";

const EVAL_FILTER_KEYS = ["purpose", "eval.name", "eval.id", "eval.leg"];

function decodeLabelSelectors(
  values: readonly string[],
): RunMetadataLabelSelector[] {
  return normalizeRunMetadataLabelSelectors(
    values.map((value) => {
      const separator = value.indexOf("=");
      if (separator <= 0) {
        throw new TypeError(
          "Every Run metadata label filter must use key=value.",
        );
      }
      return {
        key: value.slice(0, separator),
        value: value.slice(separator + 1),
      };
    }),
  );
}

function safelyDecodeLabelSelectors(values: readonly string[]): {
  selectors: RunMetadataLabelSelector[];
  error?: string;
} {
  try {
    return { selectors: decodeLabelSelectors(values) };
  } catch (error) {
    return {
      selectors: [],
      error:
        error instanceof Error
          ? error.message
          : "Run metadata label filters are invalid.",
    };
  }
}

function selectorToken(selector: RunMetadataLabelSelector): string {
  return `${selector.key}=${selector.value}`;
}

function firstSelectorValue(
  selectors: readonly RunMetadataLabelSelector[],
  key: string,
): string {
  return selectors.find((selector) => selector.key === key)?.value ?? "";
}

function RunLabelFilters({
  selectors,
  parseError,
  onReplace,
}: {
  selectors: readonly RunMetadataLabelSelector[];
  parseError: string | undefined;
  onReplace: (selectors: readonly RunMetadataLabelSelector[]) => void;
}) {
  const [validationError, setValidationError] = useState<string | undefined>();
  const [filtersOpen, setFiltersOpen] = useState(
    selectors.length > 0 || parseError !== undefined,
  );
  const fingerprint = selectors.map(selectorToken).join("\u0000");

  function addExact(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    try {
      onReplace([
        ...selectors,
        {
          key: String(data.get("labelKey") ?? ""),
          value: String(data.get("labelValue") ?? ""),
        },
      ]);
      setValidationError(undefined);
      form.reset();
    } catch (error) {
      setValidationError(
        error instanceof Error ? error.message : "Label filter is invalid.",
      );
    }
  }

  function applyEval(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const next = selectors.filter(
      (selector) => !EVAL_FILTER_KEYS.includes(selector.key),
    );
    next.push({ key: "purpose", value: "eval" });
    for (const [key, field] of [
      ["eval.name", "evalName"],
      ["eval.id", "evalID"],
      ["eval.leg", "evalLeg"],
    ] as const) {
      const value = String(data.get(field) ?? "");
      if (value !== "") {
        next.push({ key, value });
      }
    }
    try {
      onReplace(next);
      setValidationError(undefined);
    } catch (error) {
      setValidationError(
        error instanceof Error ? error.message : "Eval filter is invalid.",
      );
    }
  }

  function remove(selector: RunMetadataLabelSelector): void {
    onReplace(
      selectors.filter(
        (candidate) =>
          candidate.key !== selector.key || candidate.value !== selector.value,
      ),
    );
    setValidationError(undefined);
  }

  return (
    <details
      className="run-label-filters"
      open={filtersOpen}
      onToggle={(event) => setFiltersOpen(event.currentTarget.open)}
    >
      <summary>
        <span className="run-label-filter-title">
          <strong>Metadata &amp; eval filters</strong>
          <small>Exact, URL-synced selectors</small>
        </span>
        <span
          className={`run-label-filter-count ${selectors.length > 0 ? "has-active" : ""}`}
          aria-label={`${selectors.length} ${selectors.length === 1 ? "active filter" : "active filters"}`}
        >
          <strong>{selectors.length}</strong>
          <small>
            {selectors.length === 1 ? "active filter" : "active filters"}
          </small>
        </span>
      </summary>
      <div className="run-label-filter-body">
        <p className="muted-copy">
          Every active key=value pair must match. Values are exact and may
          contain “=”. Filters are URL-visible; do not use labels for secrets.
        </p>
        <form
          className="run-eval-filter"
          key={fingerprint}
          onSubmit={applyEval}
        >
          <label>
            Eval name
            <input
              name="evalName"
              type="text"
              defaultValue={firstSelectorValue(selectors, "eval.name")}
            />
          </label>
          <label>
            Eval ID
            <input
              name="evalID"
              type="text"
              defaultValue={firstSelectorValue(selectors, "eval.id")}
            />
          </label>
          <label>
            Eval leg
            <input
              name="evalLeg"
              type="text"
              defaultValue={firstSelectorValue(selectors, "eval.leg")}
            />
          </label>
          <button className="secondary-button" type="submit">
            Apply eval filters
          </button>
        </form>
        <form className="run-exact-label-filter" onSubmit={addExact}>
          <label>
            Exact label key
            <input name="labelKey" type="text" autoComplete="off" />
          </label>
          <label>
            Exact label value
            <input name="labelValue" type="text" autoComplete="off" />
          </label>
          <button
            className="secondary-button"
            type="submit"
            disabled={selectors.length >= RUN_METADATA_LABEL_LIMIT}
          >
            Add exact filter
          </button>
        </form>
        {parseError === undefined && validationError === undefined ? null : (
          <p className="field-error" role="alert">
            {parseError ?? validationError}
          </p>
        )}
        {parseError === undefined ? null : (
          <button
            className="secondary-button run-label-filter-recovery"
            type="button"
            onClick={() => onReplace([])}
          >
            Clear malformed metadata filters
          </button>
        )}
        {selectors.length === 0 ? (
          <p className="run-label-filter-empty">
            No metadata filters applied — all Runs are included.
          </p>
        ) : (
          <div className="run-active-label-filters">
            {selectors.map((selector) => (
              <span
                className="run-active-label-filter"
                key={selectorToken(selector)}
              >
                <code>{selectorToken(selector)}</code>
                <button
                  type="button"
                  aria-label={`Remove filter ${selectorToken(selector)}`}
                  onClick={() => remove(selector)}
                >
                  ×
                </button>
              </span>
            ))}
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                onReplace([]);
                setValidationError(undefined);
              }}
            >
              Clear metadata filters
            </button>
          </div>
        )}
      </div>
    </details>
  );
}

export function RunListRoute() {
  const api = usePublicAPI();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedState = searchParams.get("state");
  const state = RUN_STATES.find((candidate) => candidate === requestedState) as
    WorkflowRunState | undefined;
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const cursor = cursors.at(-1);
  const encodedLabelSelectors = searchParams.getAll("label");
  const decoded = safelyDecodeLabelSelectors(encodedLabelSelectors);
  const selectorTokens = decoded.selectors.map(selectorToken);
  const query = useQuery({
    queryKey: queryKeys.runs.list(state, cursor, selectorTokens),
    queryFn: () =>
      listRuns(api, {
        ...(state === undefined ? {} : { state }),
        ...(cursor === undefined ? {} : { cursor }),
        labelSelectors: decoded.selectors,
      }),
    enabled: decoded.error === undefined,
  });

  function replaceLabelSelectors(
    nextSelectors: readonly RunMetadataLabelSelector[],
  ): void {
    const normalized = normalizeRunMetadataLabelSelectors(nextSelectors);
    const next = new URLSearchParams(searchParams);
    next.delete("label");
    for (const selector of normalized) {
      next.append("label", selectorToken(selector));
    }
    setSearchParams(next, { replace: true });
    setCursors([undefined]);
  }

  return (
    <section className="route-page runs-page">
      <header className="route-header-row">
        <div>
          <p className="eyebrow">Authoritative execution history</p>
          <h2>Runs</h2>
          <p className="lede">
            Lifecycle state comes only from Go Server snapshots. Open a Run to
            inspect its ordered Stage attempts and live typed Planner plan.
          </p>
        </div>
        <button
          className="secondary-button"
          type="button"
          disabled={query.isFetching || decoded.error !== undefined}
          onClick={() => void query.refetch()}
        >
          {query.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      <div className="panel run-library">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Owner scope</p>
            <h3>Workflow Runs</h3>
          </div>
          <label className="compact-select">
            State
            <select
              value={state ?? ""}
              onChange={(event) => {
                const selected = event.target.value as WorkflowRunState | "";
                const next = new URLSearchParams(searchParams);
                if (selected === "") {
                  next.delete("state");
                } else {
                  next.set("state", selected);
                }
                setSearchParams(next, { replace: true });
                setCursors([undefined]);
              }}
            >
              <option value="">All states</option>
              {RUN_STATES.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </select>
          </label>
        </div>
        <RunLabelFilters
          key={encodedLabelSelectors.join("\u0000")}
          selectors={decoded.selectors}
          parseError={decoded.error}
          onReplace={replaceLabelSelectors}
        />
        {decoded.error !== undefined ? (
          <div className="compact-empty">
            Clear the malformed metadata filters to load Runs.
          </div>
        ) : query.isPending ? (
          <p className="loading-copy" aria-live="polite">
            Loading Runs…
          </p>
        ) : query.error !== null ? (
          <ErrorNotice error={query.error} />
        ) : query.data.items.length === 0 ? (
          <div className="compact-empty">
            <strong>No Runs match this view.</strong>
            <p>Create one from an exact published Workflow.</p>
          </div>
        ) : (
          <div className="table-scroll">
            <table className="responsive-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Workflow</th>
                  <th>State</th>
                  <th>Run metadata labels</th>
                  <th>Created</th>
                  <th>Updated</th>
                  <th>Finished</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((run) => (
                  <tr key={run.runId}>
                    <td data-label="Run">
                      <Link to={`/runs/${encodeURIComponent(run.runId)}`}>
                        {run.runId}
                      </Link>
                    </td>
                    <td data-label="Workflow">
                      <code>{run.workflow}</code>
                    </td>
                    <td data-label="State">
                      <StateBadge state={run.state} />
                    </td>
                    <td data-label="Run metadata labels">
                      <RunMetadataLabelChips labels={run.labels} />
                    </td>
                    <td data-label="Created">
                      {formatTimestamp(run.createdAt)}
                    </td>
                    <td data-label="Updated">
                      {formatTimestamp(run.updatedAt)}
                    </td>
                    <td data-label="Finished">
                      {run.finishedAt === undefined
                        ? "—"
                        : formatTimestamp(run.finishedAt)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <CursorControls
          label="Run pages"
          canGoBack={cursors.length > 1}
          {...(query.data?.page.hasMore === true &&
          query.data.page.nextCursor !== undefined
            ? { nextCursor: query.data.page.nextCursor }
            : {})}
          onBack={() =>
            setCursors((current) =>
              current.slice(0, Math.max(1, current.length - 1)),
            )
          }
          onNext={(next) => setCursors((current) => [...current, next])}
        />
      </div>
    </section>
  );
}
