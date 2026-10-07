import { useQuery } from "@tanstack/react-query";
import { useId } from "react";
import { Link, useNavigate, useSearchParams, type To } from "react-router";
import { usePublicAPI } from "../../api/context";
import {
  EVAL_STATES,
  listEvalExperiments,
  type EvalExperiment,
  type EvalListQuery,
} from "../../api/evals";
import { queryKeys } from "../../api/query-keys";
import { useDocumentTitle } from "../../app/document-title";
import { RecordedTime } from "../../app/recorded-time";
import {
  DetailHeader,
  DetailPane,
  EmptyState,
  IdChip,
  Kbd,
  ListPane,
  ListRow,
  ListSection,
  PaneLayout,
  StatusChip,
  StatusGlyph,
  useListNavigation,
} from "../../ui";
import { ArmKey, EvalError, EvalField, EvalStateChip } from "./common";
import { EvalDiagnostics } from "./diagnostics";
import {
  controlModeLabel,
  EVAL_SECTIONS,
  EVAL_STATE_LABELS,
  evalConclusionLabel,
  evalStateLabel,
  executionKindLabel,
  expectedMembersText,
  experimentLanding,
  experimentPath,
} from "./labels";
import { EvalOverviewSummary } from "./overview";
import { evalListPollInterval } from "./polling";
import { useEvalExperiment, useEvalProjects } from "./queries";

type EvalListItem = Awaited<
  ReturnType<typeof listEvalExperiments>
>["items"][number];

const FILTER_KEYS = ["state", "controlMode", "projectId", "datasetId"] as const;

/**
 * The experiments list (V3B pane frame). Filters, the page cursor and the
 * selected experiment (`?experiment=`) live in the URL; the detail pane
 * previews the selected experiment and opens its pages.
 */
export function EvalListRoute() {
  useDocumentTitle("Evals");
  const api = usePublicAPI(),
    projects = useEvalProjects(),
    navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const state = params.get("state"),
    mode = params.get("controlMode"),
    cursor = params.get("cursor"),
    projectId = params.get("projectId"),
    datasetId = params.get("datasetId"),
    selectedId = params.get("experiment") || undefined;
  const query: EvalListQuery = {
    ...(EVAL_STATES.some((x) => x === state)
      ? { state: state as NonNullable<EvalListQuery["state"]> }
      : {}),
    ...(mode === "server" || mode === "external" ? { controlMode: mode } : {}),
    ...(cursor ? { cursor } : {}),
    ...(projectId ? { projectId } : {}),
    ...(datasetId ? { datasetId } : {}),
  };
  const list = useQuery({
    queryKey: queryKeys.evals.list(query),
    queryFn: ({ signal }) => listEvalExperiments(api, query, signal),
    refetchInterval: (loaded) =>
      evalListPollInterval(loaded.state.data?.items, !!cursor),
  });
  const items = list.data?.items ?? [];
  const filtered = FILTER_KEYS.some((key) => params.get(key));
  const paged = !!cursor || !!list.data?.page.hasMore;

  function filter(key: string, value: string) {
    const next = new URLSearchParams(params);
    next.delete("cursor");
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next);
  }
  function selection(experimentId: string | undefined): To {
    const next = new URLSearchParams(params);
    if (experimentId === undefined) next.delete("experiment");
    else next.set("experiment", experimentId);
    const search = next.toString();
    return { pathname: "/evals", search: search ? `?${search}` : "" };
  }
  const selectedItem = items.find((item) => item.experimentId === selectedId);
  const { containerProps } = useListNavigation({
    count: items.length,
    index: items.findIndex((item) => item.experimentId === selectedId),
    onMove: (index) => {
      const item = items[index];
      if (item) void navigate(selection(item.experimentId));
    },
    onOpen: (index) => {
      const item = items[index];
      if (item) void navigate(experimentLanding(item));
    },
  });

  const listPane = (
    <ListPane
      header={
        <header className="ui-list-pane-header eval-list-header">
          <div className="eval-list-title-row">
            <h1 className="ui-list-pane-title">Experiments</h1>
            <Link
              className="ui-btn"
              data-variant="primary"
              data-size="sm"
              to="/evals/new"
            >
              New experiment
            </Link>
          </div>
          <p className="ui-list-pane-subtitle">
            Compare execution quality, cost and coverage across two variants.
          </p>
        </header>
      }
      toolbar={
        <div className="eval-list-toolbar">
          <nav className="eval-collections" aria-label="Evaluation collections">
            <Link to="/evals/datasets">Datasets</Link>
            <Link to="/evals/legacy">Legacy evaluation workspaces</Link>
          </nav>
          <details className="eval-filters" open={filtered}>
            <summary>
              <span>Filter experiments{filtered ? " · active" : ""}</span>
            </summary>
            <div className="eval-filter-fields">
              <EvalField label="Lifecycle">
                <select
                  value={state ?? ""}
                  onChange={(e) => filter("state", e.target.value)}
                >
                  <option value="">All states</option>
                  {EVAL_STATES.map((s) => (
                    <option key={s} value={s}>
                      {EVAL_STATE_LABELS[s].label}
                    </option>
                  ))}
                </select>
              </EvalField>
              <EvalField label="Control mode">
                <select
                  value={mode ?? ""}
                  onChange={(e) => filter("controlMode", e.target.value)}
                >
                  <option value="">All control modes</option>
                  <option value="server">{controlModeLabel("server")}</option>
                  <option value="external">
                    {controlModeLabel("external")}
                  </option>
                </select>
              </EvalField>
              <EvalField label="Evaluation workspace">
                <select
                  value={projectId ?? ""}
                  onChange={(e) => filter("projectId", e.target.value)}
                >
                  <option value="">All workspaces</option>
                  {projects.data?.map((p) => (
                    <option key={p.projectId} value={p.projectId}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </EvalField>
              <EvalField label="Dataset ID">
                <input
                  value={datasetId ?? ""}
                  onChange={(e) => filter("datasetId", e.target.value)}
                />
              </EvalField>
            </div>
            {filtered ? (
              <button
                type="button"
                className="ui-btn"
                data-size="sm"
                onClick={() =>
                  setParams(selectedId ? { experiment: selectedId } : {})
                }
              >
                Clear filters
              </button>
            ) : null}
          </details>
        </div>
      }
      footer={
        paged || items.length > 0 ? (
          <>
            {paged ? (
              <nav className="eval-pages" aria-label="Experiment pages">
                <button
                  type="button"
                  className="ui-btn"
                  data-size="xs"
                  disabled={!cursor}
                  onClick={() => filter("cursor", "")}
                >
                  First page
                </button>
                <button
                  type="button"
                  className="ui-btn"
                  data-size="xs"
                  disabled={!list.data?.page.hasMore}
                  onClick={() => {
                    if (list.data?.page.nextCursor) {
                      const next = new URLSearchParams(params);
                      next.set("cursor", list.data.page.nextCursor);
                      setParams(next);
                    }
                  }}
                >
                  Next page
                </button>
              </nav>
            ) : null}
            {items.length > 0 ? (
              <span className="eval-key-hints">
                <Kbd>J</Kbd> <Kbd>K</Kbd> move · <Kbd>Enter</Kbd> open
              </span>
            ) : null}
          </>
        ) : undefined
      }
    >
      <EvalError
        error={list.error ?? projects.error}
        reload={() => {
          filter("cursor", "");
          void list.refetch();
          void projects.refetch();
        }}
      />
      {list.isPending ? (
        <p className="eval-pane-note" role="status">
          Loading experiments…
        </p>
      ) : null}
      {list.data?.items.length === 0 ? (
        <EmptyState title="No matching experiments">
          <p>
            Create an experiment or change the filters. Earlier evaluation
            workspaces remain in legacy history.
          </p>
        </EmptyState>
      ) : null}
      {items.length > 0 ? (
        <div {...containerProps}>
          <ListSection>
            {items.map((item) => (
              <ExperimentRow
                key={item.experimentId}
                item={item}
                to={selection(item.experimentId)}
                selected={item.experimentId === selectedId}
              />
            ))}
          </ListSection>
        </div>
      ) : null}
    </ListPane>
  );

  return (
    <PaneLayout
      listLabel="Experiments"
      detailLabel="Experiment"
      showDetail={selectedId !== undefined}
      backLink={{ to: selection(undefined), label: "Back to experiments" }}
      list={listPane}
      detail={
        selectedId === undefined ? (
          <DetailPane>
            <EmptyState title="Choose an experiment">
              <p>
                Each experiment compares a baseline (A) and a candidate (B) on
                the same cases and repetitions. Choose one to see its state,
                variants and coverage, then open its overview, comparison,
                attempts or setup.
              </p>
            </EmptyState>
          </DetailPane>
        ) : (
          <ExperimentPreview
            key={selectedId}
            experimentId={selectedId}
            item={selectedItem}
          />
        )
      }
    />
  );
}

function ExperimentRow({
  item,
  to,
  selected,
}: {
  item: EvalListItem;
  to: To;
  selected: boolean;
}) {
  const state = evalStateLabel(item.state);
  const conclusion = item.summary
    ? evalConclusionLabel(item.summary.conclusion)
    : undefined;
  return (
    <ListRow
      to={to}
      selected={selected}
      glyph={<StatusGlyph tone={state.tone} />}
      title={item.name}
      meta={
        <>
          <span className="eval-row-line">
            <span>
              <strong>{state.label}</strong> ·{" "}
              {executionKindLabel(item.executionKind)} ·{" "}
              {controlModeLabel(item.controlMode)} ·{" "}
              {expectedMembersText(item.expectedMembers)}
            </span>
          </span>
          {item.variants?.length ? (
            <span className="eval-row-line eval-row-variants">
              {item.variants.map((variant) => (
                <span className="eval-row-variant" key={variant.id}>
                  <span className="eval-variant-id">{variant.id}</span>{" "}
                  <code>{variant.selector}</code>
                </span>
              ))}
            </span>
          ) : null}
          <span className="eval-row-line">
            {conclusion ? (
              <StatusChip tone={conclusion.tone} size="sm">
                {conclusion.label}
              </StatusChip>
            ) : (
              <span>
                {item.state === "draft"
                  ? "Configure the variants and cases, then prepare the experiment."
                  : "Comparison pending"}
              </span>
            )}
            <span>
              Updated <RecordedTime value={item.updatedAt} />
            </span>
          </span>
        </>
      }
    />
  );
}

function variantSelector(experiment: EvalExperiment, variantId: string) {
  return (experiment.setup ?? experiment.draft)?.variants.find(
    (variant) => variant.id === variantId,
  )?.selector;
}

/** The selected experiment: state, matrix, variants, conclusion, coverage. */
function ExperimentPreview({
  experimentId,
  item,
}: {
  experimentId: string;
  item: EvalListItem | undefined;
}) {
  const experiment = useEvalExperiment(experimentId);
  const data = experiment.data;
  const name = data?.name ?? item?.name ?? "Experiment";
  const state = data?.state ?? item?.state;
  const kind = data?.executionKind ?? item?.executionKind;
  const mode = data?.controlMode ?? item?.controlMode;
  const landing =
    state === undefined
      ? experimentPath(experimentId, "overview")
      : experimentLanding({ experimentId, state });
  return (
    <DetailPane
      header={
        <DetailHeader
          title={name}
          status={
            state === undefined ? undefined : (
              <EvalStateChip state={state} size="sm" />
            )
          }
          meta={
            <>
              {kind && mode ? (
                <span>
                  {executionKindLabel(kind)} · {controlModeLabel(mode)}
                </span>
              ) : null}
              <IdChip value={experimentId} label="experiment ID" />
            </>
          }
          actions={
            <Link
              className="ui-btn"
              data-variant="primary"
              data-size="sm"
              to={landing}
            >
              {state === "draft" ? "Continue setup" : "Open experiment"}
            </Link>
          }
        />
      }
    >
      <EvalError
        error={experiment.error}
        reload={() => void experiment.refetch()}
      />
      {data ? (
        <PreviewBody experiment={data} />
      ) : experiment.isPending ? (
        <p role="status">Loading experiment…</p>
      ) : null}
    </DetailPane>
  );
}

function PreviewBody({ experiment }: { experiment: EvalExperiment }) {
  const variantsHeading = useId();
  const setup = experiment.setup ?? experiment.draft;
  const comparison = setup?.comparison;
  const cases = setup?.caseIds?.length;
  const repetitions = setup?.repetitions;
  return (
    <div className="eval-preview">
      <EvalDiagnostics experiment={experiment} />
      <dl className="eval-glance" aria-label="At a glance">
        <div>
          <dt>Matrix</dt>
          <dd>
            {cases !== undefined && repetitions !== undefined
              ? `${cases} cases × 2 variants × ${repetitions} repetitions = `
              : ""}
            {expectedMembersText(experiment.expectedMembers)}
          </dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>
            {setup?.dataset?.id ? (
              <>
                {setup.dataset.id}{" "}
                <IdChip
                  value={setup.dataset.revision}
                  label="dataset revision"
                />
              </>
            ) : experiment.setup?.source ? (
              `${experiment.setup.source.system}/${experiment.setup.source.id}`
            ) : (
              "Not selected yet"
            )}
          </dd>
        </div>
        <div>
          <dt>Updated</dt>
          <dd>
            <RecordedTime value={experiment.updatedAt} />
          </dd>
        </div>
      </dl>
      {comparison ? (
        <section className="eval-block" aria-labelledby={variantsHeading}>
          <h3 id={variantsHeading}>Variants</h3>
          <ul className="eval-variant-list" role="list">
            {(
              [
                ["a", comparison.baseline, "A · Baseline"],
                ["b", comparison.candidate, "B · Candidate"],
              ] as const
            ).map(([arm, id, label]) => {
              const selector = variantSelector(experiment, id);
              return (
                <li key={arm}>
                  <ArmKey arm={arm}>{label}</ArmKey>
                  {selector ? (
                    <IdChip
                      value={selector}
                      display={selector}
                      label={`${arm.toUpperCase()} version`}
                    />
                  ) : (
                    <span className="eval-muted">Not chosen yet</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}
      {experiment.state === "draft" ? (
        <p className="eval-muted">
          Configure the variants and cases, then prepare the experiment.
        </p>
      ) : (
        <EvalOverviewSummary experiment={experiment} />
      )}
      <nav
        className="eval-preview-links"
        aria-label="Open an experiment section"
      >
        {EVAL_SECTIONS.map((section) => (
          <Link
            key={section.value}
            className="ui-btn"
            data-size="sm"
            to={experimentPath(experiment.experimentId, section.value)}
          >
            {section.label}
          </Link>
        ))}
      </nav>
    </div>
  );
}
