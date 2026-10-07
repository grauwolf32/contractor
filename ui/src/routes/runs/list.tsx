import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useId, useState } from "react";
import { useSearchParams } from "react-router";

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
  type RunSummary,
} from "../../api/runs";
import { CursorControls } from "../../app/cursor-controls";
import { useURLCursorStack } from "../../app/pagination";
import { ErrorNotice } from "../../app/error-notice";
import { formatTimestamp } from "../../app/format";
import { RefreshButton } from "../../app/refresh-button";
import { RecordedTime } from "../../app/recorded-time";
import { QueryView } from "../../app/query-view";
import { ConfirmRemovalDialog } from "../../app/confirm-removal-dialog";
import { DeleteIcon } from "../../app/delete-icon";
import { Icon } from "../../app/icon";
import { EmptyState, IdChip } from "../../ui";
import {
  DisclosureChevron,
  RunMetadataLabelChips,
  RunStateChip,
} from "./components";
import { RunIdLink, RunsFilter } from "./list-parts";
import { RUN_STATE_LABELS } from "./run-state";

const EVAL_FILTER_KEYS = ["purpose", "eval.name", "eval.id", "eval.leg"];

const STATE_OPTIONS = [
  { value: "" as const, label: "All" },
  ...TERMINAL_RUN_STATES.map((state) => ({
    value: state,
    label: RUN_STATE_LABELS[state].label,
  })),
];

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

function selectorLabel(selector: RunMetadataLabelSelector): string {
  return selector.value === ""
    ? selector.key
    : `${selector.key}:${selector.value}`;
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
      className="run-label-filters runs-label-filters"
      open={filtersOpen}
      onToggle={(event) => setFiltersOpen(event.currentTarget.open)}
    >
      <summary className="runs-label-filters-summary">
        <DisclosureChevron />
        <span className="runs-label-filters-title">
          <strong>Metadata &amp; eval filters</strong>
          <small>Find Runs by their labels</small>
        </span>
        <span
          className="runs-label-filters-count"
          data-active={selectors.length > 0 ? "" : undefined}
        >
          {selectors.length}{" "}
          {selectors.length === 1 ? "active filter" : "active filters"}
        </span>
      </summary>
      <div className="runs-label-filters-body">
        <p className="runs-hint">
          Every active key=value pair must match. Values are exact and may
          contain “=”. Filters are URL-visible; do not use labels for secrets.
        </p>
        <form
          className="runs-filter-form"
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
          <button className="ui-btn" data-size="sm" type="submit">
            Apply eval filters
          </button>
        </form>
        <form className="runs-filter-form" onSubmit={addExact}>
          <label>
            Label key
            <input name="labelKey" type="text" autoComplete="off" />
          </label>
          <label>
            Label value
            <input name="labelValue" type="text" autoComplete="off" />
          </label>
          <button
            className="ui-btn"
            data-size="sm"
            type="submit"
            disabled={selectors.length >= RUN_METADATA_LABEL_LIMIT}
          >
            Add filter
          </button>
        </form>
        {parseError === undefined && validationError === undefined ? null : (
          <p className="runs-field-error" role="alert">
            {parseError ?? validationError}
          </p>
        )}
        {parseError === undefined ? null : (
          <button
            className="ui-btn"
            data-size="sm"
            type="button"
            onClick={() => onReplace([])}
          >
            Clear malformed metadata filters
          </button>
        )}
        {selectors.length === 0 ? (
          <p className="runs-hint">
            No metadata filters applied — all Runs are included.
          </p>
        ) : (
          <div className="runs-active-filters">
            {selectors.map((selector) => (
              <span
                className="runs-active-filter"
                key={selectorToken(selector)}
              >
                <code>{selectorLabel(selector)}</code>
                <button
                  type="button"
                  aria-label={`Remove filter ${selectorLabel(selector)}`}
                  onClick={() => remove(selector)}
                >
                  <Icon name="close" />
                </button>
              </span>
            ))}
            <button
              className="ui-btn"
              data-size="xs"
              data-variant="ghost"
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

function contextSummary(labels: Readonly<Record<string, string>>): string {
  const count = Object.keys(labels).length;
  return (
    labels["eval.name"] ??
    labels.purpose ??
    (count === 0 ? "Details" : `${count} ${count === 1 ? "label" : "labels"}`)
  );
}

function CompletedRunRow({
  run,
  onDelete,
}: {
  run: RunSummary;
  onDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const contextId = useId();
  return (
    <>
      <tr data-expanded={expanded ? "" : undefined}>
        <td className="run-list-id-cell">
          <RunIdLink runId={run.runId} returnLabel="Completed" />
        </td>
        <td className="run-list-workflow-cell">
          <span className="runs-cell-label">Workflow</span>
          <code>{run.workflow}</code>
        </td>
        <td className="run-list-state-cell">
          <RunStateChip state={run.state} />
        </td>
        <td className="run-list-labels-cell">
          <button
            type="button"
            className="runs-context-toggle"
            aria-label={`Context for ${run.runId}`}
            aria-expanded={expanded}
            aria-controls={contextId}
            onClick={() => setExpanded((value) => !value)}
          >
            <DisclosureChevron />
            <span>{contextSummary(run.labels)}</span>
          </button>
        </td>
        <td className="run-list-finished-cell">
          <span className="runs-cell-label">Finished</span>
          {run.finishedAt === undefined ? (
            "—"
          ) : (
            <RecordedTime value={run.finishedAt} />
          )}
        </td>
        <td className="run-list-actions-cell">
          {run.deletable === true ? (
            <button
              className="ui-btn runs-delete"
              data-size="sm"
              data-variant="ghost"
              type="button"
              aria-label={`Delete Run ${run.runId}`}
              title="Delete completed Run"
              onClick={onDelete}
            >
              <DeleteIcon />
            </button>
          ) : null}
        </td>
      </tr>
      {expanded ? (
        <tr className="runs-context-row">
          <td colSpan={6}>
            <section
              className="runs-context-details"
              id={contextId}
              aria-label={`Context for ${run.runId}`}
            >
              <RunMetadataLabelChips
                labels={run.labels}
                empty="No metadata labels."
              />
              <dl className="runs-inline-facts">
                <div>
                  <dt>Run ID</dt>
                  <dd>
                    <IdChip value={run.runId} label="Run ID" />
                  </dd>
                </div>
                <div>
                  <dt>Created</dt>
                  <dd>{formatTimestamp(run.createdAt)}</dd>
                </div>
                <div>
                  <dt>Updated</dt>
                  <dd>{formatTimestamp(run.updatedAt)}</dd>
                </div>
              </dl>
            </section>
          </td>
        </tr>
      ) : null}
    </>
  );
}

/**
 * Runs → Completed: terminal Runs newest first (S18), filtered by state and
 * exact metadata labels, with deletion for Runs the Server marks deletable.
 * Filters and the page cursor live in the URL.
 */
export function CompletedRunsPanel() {
  const api = usePublicAPI();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedState = searchParams.get("state");
  const state = TERMINAL_RUN_STATES.find(
    (candidate) => candidate === requestedState,
  ) as WorkflowRunState | undefined;
  const pages = useURLCursorStack();
  const [deleteTarget, setDeleteTarget] = useState<string | undefined>();
  const cursor = pages.cursor;
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

  function replaceState(selected: WorkflowRunState | ""): void {
    const next = new URLSearchParams(searchParams);
    next.delete("cursor");
    if (selected === "") {
      next.delete("state");
    } else {
      next.set("state", selected);
    }
    setSearchParams(next, { replace: true });
  }

  function replaceLabelSelectors(
    nextSelectors: readonly RunMetadataLabelSelector[],
  ): void {
    const normalized = normalizeRunMetadataLabelSelectors(nextSelectors);
    const next = new URLSearchParams(searchParams);
    next.delete("cursor");
    next.delete("label");
    for (const selector of normalized) {
      next.append("label", selectorToken(selector));
    }
    setSearchParams(next, { replace: true });
  }

  return (
    <div className="runs-view">
      <div className="runs-toolbar">
        <div className="runs-filters">
          <RunsFilter
            label="State"
            options={STATE_OPTIONS}
            value={state ?? ""}
            onChange={replaceState}
          />
        </div>
        <RefreshButton
          className="runs-icon-button"
          isFetching={query.isFetching}
          disabled={decoded.error !== undefined}
          onRefresh={() => void query.refetch()}
        />
      </div>
      <RunLabelFilters
        key={encodedLabelSelectors.join("\u0000")}
        selectors={decoded.selectors}
        parseError={decoded.error}
        onReplace={replaceLabelSelectors}
      />
      <div className="runs-results">
        {decoded.error !== undefined ? (
          <EmptyState title="Clear the malformed metadata filters to load Runs." />
        ) : (
          <QueryView
            query={query}
            loading={
              <p className="runs-loading" aria-live="polite">
                Loading completed Runs…
              </p>
            }
            onRetry={() => void query.refetch()}
            isEmpty={(data) => data.items.length === 0}
            empty={
              <EmptyState title="No completed Runs match this view.">
                <p>Terminal Runs appear here after execution finishes.</p>
              </EmptyState>
            }
          >
            {(data) => (
              <table className="runs-table" data-has-actions="">
                <thead>
                  <tr>
                    <th scope="col">Run</th>
                    <th scope="col">Workflow</th>
                    <th scope="col">State</th>
                    <th scope="col">Context &amp; details</th>
                    <th scope="col">Finished</th>
                    <th scope="col">
                      <span className="ui-visually-hidden">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((run) => (
                    <CompletedRunRow
                      key={run.runId}
                      run={run}
                      onDelete={() => {
                        deletion.reset();
                        setDeleteTarget(run.runId);
                      }}
                    />
                  ))}
                </tbody>
              </table>
            )}
          </QueryView>
        )}
      </div>
      <CursorControls label="Run pages" {...pages.controls(query.data?.page)} />
      {deleteTarget === undefined ? null : (
        <ConfirmRemovalDialog
          className="run-delete-dialog"
          eyebrow="Permanent action"
          title="Delete completed Run?"
          description={
            <>
              This permanently removes Run <code>{deleteTarget}</code>, its
              execution history, and all Run-owned Artifacts. Shared source
              Artifacts and published Project outputs are retained.
            </>
          }
          confirmLabel="Delete Run"
          pendingLabel="Deleting…"
          pending={deletion.isPending}
          dismissOnBackdrop={false}
          error={
            deletion.error === null ? null : (
              <ErrorNotice error={deletion.error} />
            )
          }
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
