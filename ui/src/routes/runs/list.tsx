import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useEffect, useId, useState } from "react";
import { Link, useSearchParams } from "react-router";

import { usePublicAPI } from "../../api/context";
import { queryKeys } from "../../api/query-keys";
import {
  normalizeRunMetadataLabelSelectors,
  RUN_METADATA_LABEL_LIMIT,
  type RunMetadataLabelSelector,
} from "../../api/run-metadata-labels";
import {
  deleteRun,
  listRuns,
  TERMINAL_RUN_STATES,
  type WorkflowRunState,
} from "../../api/runs";
import {
  CursorControls,
  ErrorNotice,
  formatTimestamp,
} from "../artifacts/common";
import { RunMetadataLabelChips, StateBadge } from "./components";

const EVAL_FILTER_KEYS = ["purpose", "eval.name", "eval.id", "eval.leg"];

function DeleteRunDialog({
  runId,
  pending,
  error,
  onCancel,
  onConfirm,
}: {
  runId: string;
  pending: boolean;
  error: Error | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent): void {
      if (event.key === "Escape" && !pending) {
        onCancel();
      }
    }
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onCancel, pending]);

  return (
    <div className="project-dialog-backdrop" role="presentation">
      <section
        className="project-dialog run-delete-dialog panel"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={heading}
      >
        <div className="project-dialog-heading">
          <div>
            <p className="eyebrow">Permanent action</p>
            <h2 id={heading}>Delete completed Run?</h2>
          </div>
        </div>
        <p>
          This permanently removes Run <code>{runId}</code>, its execution
          history, and all Run-owned Artifacts. Shared source Artifacts and
          published Project outputs are retained.
        </p>
        {error === null ? null : <ErrorNotice error={error} />}
        <div className="run-delete-dialog-actions">
          <button
            className="secondary-button"
            type="button"
            autoFocus
            disabled={pending}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="danger-button"
            type="button"
            disabled={pending}
            onClick={onConfirm}
          >
            {pending ? "Deleting…" : "Delete Run"}
          </button>
        </div>
      </section>
    </div>
  );
}

function compactRunId(runId: string): string {
  return runId.length <= 24
    ? runId
    : `${runId.slice(0, 12)}…${runId.slice(-8)}`;
}

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

export function CompletedRunsPanel() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedState = searchParams.get("state");
  const state = TERMINAL_RUN_STATES.find(
    (candidate) => candidate === requestedState,
  ) as WorkflowRunState | undefined;
  const [cursors, setCursors] = useState<Array<string | undefined>>([
    undefined,
  ]);
  const [deleteTarget, setDeleteTarget] = useState<string | undefined>();
  const cursor = cursors.at(-1);
  const encodedLabelSelectors = searchParams.getAll("label");
  const decoded = safelyDecodeLabelSelectors(encodedLabelSelectors);
  const selectorTokens = decoded.selectors.map(selectorToken);
  const query = useQuery({
    queryKey: queryKeys.runs.list(state, cursor, selectorTokens, "terminal"),
    queryFn: () =>
      listRuns(api, {
        ...(state === undefined ? {} : { state }),
        lifecycle: "terminal",
        ...(cursor === undefined ? {} : { cursor }),
        labelSelectors: decoded.selectors,
      }),
    enabled: decoded.error === undefined,
  });
  const deletion = useMutation({
    mutationFn: (runId: string) => deleteRun(api, runId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: queryKeys.runs.all }),
        queryClient.invalidateQueries({ queryKey: queryKeys.projects.all }),
      ]);
      setDeleteTarget(undefined);
    },
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
    <div className="panel run-library run-view-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Owner scope</p>
          <h3>Completed Runs</h3>
        </div>
        <div className="run-view-controls">
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
              <option value="">All terminal states</option>
              {TERMINAL_RUN_STATES.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </select>
          </label>
          <button
            className="secondary-button"
            type="button"
            disabled={query.isFetching || decoded.error !== undefined}
            onClick={() => void query.refetch()}
          >
            {query.isFetching ? "Refreshing…" : "Refresh"}
          </button>
        </div>
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
          Loading completed Runs…
        </p>
      ) : query.error !== null ? (
        <ErrorNotice error={query.error} />
      ) : query.data.items.length === 0 ? (
        <div className="compact-empty">
          <strong>No completed Runs match this view.</strong>
          <p>Terminal Runs appear here after execution finishes.</p>
        </div>
      ) : (
        <div className="table-scroll">
          <table className="responsive-table run-list-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Workflow</th>
                <th>State</th>
                <th>Run metadata labels</th>
                <th>Created</th>
                <th>Updated</th>
                <th>Finished</th>
                <th>
                  <span className="visually-hidden">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {query.data.items.map((run) => (
                <tr key={run.runId}>
                  <td className="run-list-id-cell" data-label="Run">
                    <Link
                      className="run-list-id-link"
                      to={`/runs/${encodeURIComponent(run.runId)}`}
                      aria-label={run.runId}
                      title={run.runId}
                    >
                      {compactRunId(run.runId)}
                    </Link>
                  </td>
                  <td className="run-list-workflow-cell" data-label="Workflow">
                    <span className="run-list-mobile-label">Workflow</span>
                    <code>{run.workflow}</code>
                  </td>
                  <td className="run-list-state-cell" data-label="State">
                    <StateBadge state={run.state} />
                  </td>
                  <td
                    className={`run-list-labels-cell ${Object.keys(run.labels).length === 0 ? "run-list-labels-empty" : ""}`}
                    data-label="Run metadata labels"
                  >
                    <span className="run-list-mobile-label">Labels</span>
                    <RunMetadataLabelChips labels={run.labels} />
                  </td>
                  <td className="run-list-created-cell" data-label="Created">
                    <time dateTime={run.createdAt}>
                      {formatTimestamp(run.createdAt)}
                    </time>
                  </td>
                  <td className="run-list-updated-cell" data-label="Updated">
                    <span className="run-list-mobile-label">Updated</span>
                    <time dateTime={run.updatedAt}>
                      {formatTimestamp(run.updatedAt)}
                    </time>
                  </td>
                  <td className="run-list-finished-cell" data-label="Finished">
                    {run.finishedAt === undefined ? (
                      "—"
                    ) : (
                      <time dateTime={run.finishedAt}>
                        {formatTimestamp(run.finishedAt)}
                      </time>
                    )}
                  </td>
                  <td className="run-list-actions-cell" data-label="Actions">
                    {run.deletable === true ? (
                      <button
                        className="run-delete-trigger"
                        type="button"
                        aria-label={`Delete Run ${run.runId}`}
                        title="Delete completed Run"
                        onClick={() => {
                          deletion.reset();
                          setDeleteTarget(run.runId);
                        }}
                      >
                        <svg
                          aria-hidden="true"
                          viewBox="0 0 24 24"
                          width="18"
                          height="18"
                        >
                          <path
                            d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5m4-5v5"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                    ) : null}
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
      {deleteTarget === undefined ? null : (
        <DeleteRunDialog
          runId={deleteTarget}
          pending={deletion.isPending}
          error={deletion.error}
          onCancel={() => {
            deletion.reset();
            setDeleteTarget(undefined);
          }}
          onConfirm={() => deletion.mutate(deleteTarget)}
        />
      )}
    </div>
  );
}
