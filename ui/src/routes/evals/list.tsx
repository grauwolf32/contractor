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
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import {
  Button,
  ButtonLink,
  DetailHeader,
  DetailPane,
  EmptyState,
  Facts,
  Field,
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
import { ArmKey, EvalError, EvalStateChip } from "./common";
import { EvalDiagnostics } from "./diagnostics";
import {
  controlModeLabel,
  EVAL_SECTIONS,
  EVAL_STATE_LABELS,
  evalConclusionLabel,
  evalFreshnessText,
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
    // Later pages are a snapshot the user pages through; only page one polls.
    refetchInterval: cursor ? false : evalListPollInterval,
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
            <ButtonLink variant="primary" size="sm" to="/evals/new">
              New experiment
            </ButtonLink>
          </div>
          <p className="ui-list-pane-subtitle">
            Compare execution quality, cost and coverage across two variants.
          </p>
        </header>
      }
      toolbar={
        <div className="ui-stack eval-list-toolbar" data-gap="2">
          <nav className="eval-collections" aria-label="Evaluation collections">
            <Link to="/evals/datasets">Datasets</Link>
            <Link to="/evals/legacy">Legacy evaluation workspaces</Link>
          </nav>
          <details className="eval-filters" open={filtered}>
            <summary>
              <span>Filter experiments{filtered ? " · active" : ""}</span>
            </summary>
            <div className="eval-filter-fields">
              <Field label="Lifecycle">
                {(control) => (
                  <select
                    {...control}
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
                )}
              </Field>
              <Field label="Control mode">
                {(control) => (
                  <select
                    {...control}
                    value={mode ?? ""}
                    onChange={(e) => filter("controlMode", e.target.value)}
                  >
                    <option value="">All control modes</option>
                    <option value="server">{controlModeLabel("server")}</option>
                    <option value="external">
                      {controlModeLabel("external")}
                    </option>
                  </select>
                )}
              </Field>
              <Field label="Evaluation workspace">
                {(control) => (
                  <select
                    {...control}
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
                )}
              </Field>
              <Field label="Dataset ID">
                {(control) => (
                  <input
                    {...control}
                    value={datasetId ?? ""}
                    onChange={(e) => filter("datasetId", e.target.value)}
                  />
                )}
              </Field>
            </div>
            {filtered ? (
              <Button
                size="sm"
                onClick={() =>
                  setParams(selectedId ? { experiment: selectedId } : {})
                }
              >
                Clear filters
              </Button>
            ) : null}
          </details>
        </div>
      }
      footer={
        paged || items.length > 0 ? (
          <>
            {paged ? (
              <nav className="eval-pages" aria-label="Experiment pages">
                <Button
                  size="xs"
                  disabled={!cursor}
                  onClick={() => filter("cursor", "")}
                >
                  First page
                </Button>
                <Button
                  size="xs"
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
                </Button>
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
        error={projects.error}
        reload={() => void projects.refetch()}
      />
      <QueryView
        query={list}
        loading={
          <p className="eval-pane-note" role="status">
            Loading experiments…
          </p>
        }
        onRetry={() => {
          filter("cursor", "");
          void list.refetch();
        }}
        isEmpty={(data) => data.items.length === 0}
        empty={
          <EmptyState title="No matching experiments">
            <p>
              Create an experiment or change the filters. Earlier evaluation
              workspaces remain in legacy history.
            </p>
          </EmptyState>
        }
      >
        {() => (
          <div {...containerProps}>
            <ListSection aria-keyshortcuts="J K ArrowDown ArrowUp Enter">
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
        )}
      </QueryView>
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
  // The last complete view stays visible while a newer one is built; say so
  // next to its conclusion (S30:501).
  const freshness =
    item.freshness && item.freshness !== "current"
      ? evalFreshnessText(item.freshness)
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
              <>
                <StatusChip tone={conclusion.tone} size="sm">
                  {conclusion.label}
                </StatusChip>
                {freshness ? (
                  <span
                    className="eval-conclusion-note"
                    data-stale={item.freshness === "stale" || undefined}
                  >
                    {freshness}
                  </span>
                ) : null}
              </>
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
            <ButtonLink variant="primary" size="sm" to={landing}>
              {state === "draft" ? "Continue setup" : "Open experiment"}
            </ButtonLink>
          }
        />
      }
    >
      <div className="eval-preview">
        <QueryView
          query={experiment}
          loading={<p role="status">Loading experiment…</p>}
        >
          {(loaded) => <PreviewBody experiment={loaded} />}
        </QueryView>
      </div>
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
    <>
      <EvalDiagnostics experiment={experiment} />
      <Facts
        className="eval-glance"
        aria-label="At a glance"
        items={[
          {
            term: "Matrix",
            value: `${
              cases !== undefined && repetitions !== undefined
                ? `${cases} cases × 2 variants × ${repetitions} repetitions = `
                : ""
            }${expectedMembersText(experiment.expectedMembers)}`,
          },
          {
            term: "Dataset",
            value: setup?.dataset?.id ? (
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
            ),
          },
          {
            term: "Updated",
            value: <RecordedTime value={experiment.updatedAt} />,
          },
        ]}
      />
      {comparison ? (
        <section
          className="ui-stack"
          data-gap="2"
          aria-labelledby={variantsHeading}
        >
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
                    <span className="ui-caption">Not chosen yet</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}
      {experiment.state === "draft" ? (
        <p className="ui-caption">
          Configure the variants and cases, then prepare the experiment.
        </p>
      ) : (
        <EvalOverviewSummary experiment={experiment} />
      )}
      <nav className="ui-toolbar" aria-label="Open an experiment section">
        {EVAL_SECTIONS.map((section) => (
          <ButtonLink
            key={section.value}
            size="sm"
            to={experimentPath(experiment.experimentId, section.value)}
          >
            {section.label}
          </ButtonLink>
        ))}
      </nav>
    </>
  );
}
