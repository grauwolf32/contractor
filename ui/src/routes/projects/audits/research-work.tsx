import type { components } from "../../../api/generated/public";
import { AddResearchWork } from "./add-work";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import type { Audit } from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidAPIResponse, requireData } from "../../../api/error";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { useCursorStack } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { RecordedTime } from "../../../app/recorded-time";
import { readableValue } from "../../../app/vocabulary";
import {
  Button,
  Facts,
  StatusChip,
  StatusGlyph,
  TechnicalDetails,
  useListNavigation,
  Kbd,
} from "../../../ui";
import { auditPollInterval } from "../../../api/audits";
import { taskStatus } from "./research-data";
import { useAuditProjectionRefresh } from "./projection-refresh";
import {
  type ResearchTask,
  useResearchTask,
  useResearchTasks,
} from "./research-data";
import { TaskResultView } from "./task-result-view";
import { useLLMCost } from "../../../app/llm-cost-query";
import { CostAmount, CostBasisLabel } from "../../../app/llm-cost-card";
import { ErrorNotice } from "../../../app/error-notice";

function TaskAttempts({ audit, task }: { audit: Audit; task: ResearchTask }) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(false);
  const pages = useCursorStack();
  const key = [
    ...queryKeys.audits.detail(audit.auditId),
    "attempts",
    task.document.task_id,
    pages.cursor,
  ];
  const query = useQuery({
    queryKey: key,
    enabled: open,
    queryFn: async ({ signal }) => {
      const data = requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/tasks/{taskId}/attempts", {
            signal,
            params: {
              path: { auditId: audit.auditId, taskId: task.document.task_id },
              query: {
                limit: 10,
                ...(pages.cursor ? { cursor: pages.cursor } : {}),
              },
            },
          }),
        ),
      );
      if (
        data.page.hasMore &&
        (!data.items.length ||
          !data.page.nextCursor ||
          data.page.nextCursor === pages.cursor)
      )
        throw invalidAPIResponse(200, "Invalid attempt continuation");
      return data;
    },
    refetchInterval: pollInterval<
      components["schemas"]["AuditTaskAttemptPage"]
    >({
      interval: 5000,
      settled: () => auditPollInterval([audit]) === false,
    }),
  });
  useAuditProjectionRefresh(audit, key, open);
  const executionIds =
    query.data?.items.map((attempt) => attempt.execution_id) ?? [];
  const costs = useLLMCost("audit", audit.auditId, {
    executionIds,
    enabled: open && executionIds.length > 0,
    active: auditPollInterval([audit]) !== false,
    refreshKey: `${audit.revision}:${audit.state}:${audit.outstandingRunCount}`,
  });
  return (
    <TechnicalDetails
      summary="Run attempts"
      onToggle={setOpen}
      description="Technical execution is separate from the scoped result."
    >
      {open ? (
        <QueryView query={query}>
          {(page) => (
            <>
              <ul className="research-attempts">
                {page.items.map((attempt) => {
                  const cost = costs.data?.executions.find(
                    (row) => row.executionId === attempt.execution_id,
                  );
                  return (
                    <li
                      key={attempt.attempt}
                      aria-label={`Attempt ${attempt.attempt}`}
                    >
                      <strong>Attempt {attempt.attempt}</strong> ·{" "}
                      {attempt.outcome ?? readableValue(attempt.state)}
                      {attempt.run_deleted ? (
                        <span> · source Run deleted</span>
                      ) : attempt.run_id ? (
                        <>
                          {" "}
                          ·{" "}
                          <Link
                            to={`/runs/${encodeURIComponent(attempt.run_id)}`}
                          >
                            Open Run
                          </Link>
                        </>
                      ) : (
                        <span> · not dispatched</span>
                      )}
                      <span className="research-attempt-cost">
                        LLM cost ·{" "}
                        {cost ? (
                          <>
                            <CostAmount amount={cost.amount} />
                            {cost.basis !== "none" ? (
                              <>
                                {" "}
                                · <CostBasisLabel basis={cost.basis} />
                              </>
                            ) : null}
                            {cost.coverage !== "complete" ? (
                              <>
                                {" "}
                                ·{" "}
                                {cost.coverage === "pending"
                                  ? "Updating"
                                  : readableValue(cost.coverage)}
                              </>
                            ) : null}
                          </>
                        ) : costs.isError ? (
                          "Unavailable"
                        ) : (
                          "Loading…"
                        )}
                      </span>
                      {attempt.collection_disposition ? (
                        <p className="ui-caption">
                          {readableValue(attempt.collection_disposition)}
                        </p>
                      ) : null}
                      {attempt.result || attempt.stop ? (
                        <TaskResultView
                          audit={audit}
                          task={task}
                          attempt={attempt}
                        />
                      ) : null}
                    </li>
                  );
                })}
              </ul>
              {costs.error ? (
                <ErrorNotice
                  error={costs.error}
                  context="Could not refresh Run costs"
                  onRetry={() => void costs.refetch()}
                />
              ) : null}
              <CursorControls
                label="Attempt pages"
                {...pages.controls(page.page)}
              />
            </>
          )}
        </QueryView>
      ) : null}
      {open && query.isError && pages.cursor ? (
        <Button size="sm" onClick={pages.reset}>
          Restart attempt list
        </Button>
      ) : null}
    </TechnicalDetails>
  );
}

export function ResearchTaskDetail({
  audit,
  task,
}: {
  audit: Audit;
  task: ResearchTask;
}) {
  const state = taskStatus(task);
  return (
    <article className="research-task-detail">
      <header className="research-section-head">
        <div>
          <p className="eyebrow">Accepted work</p>
          <h2>
            {task.hypothesis_snapshot?.statement ?? task.document.objective}
          </h2>
        </div>
        <StatusChip tone={state.tone}>{state.label}</StatusChip>
      </header>
      {task.hypothesis_snapshot ? (
        <>
          <p className="ui-caption">
            Exact hypothesis revision {task.hypothesis_snapshot.revision}
          </p>
          <Facts
            layout="list"
            items={Object.entries(task.hypothesis_snapshot.scope).map(
              ([term, value]) => ({ term: readableValue(term), value }),
            )}
          />
        </>
      ) : null}
      {task.gap_code ? (
        <div className="checks-notice" data-tone="warning" role="status">
          <strong>Collection gap</strong>
          <p>{readableValue(task.gap_code)}</p>
          <p>A technical gap is not a hypothesis verdict.</p>
        </div>
      ) : null}
      <TaskResultView
        key={task.result?.ref.revision ?? task.document.task_id}
        audit={audit}
        task={task}
      />
      <TaskAttempts key={task.document.task_id} audit={audit} task={task} />
      <TechnicalDetails description="Task identity, exact inputs and system-derived origin.">
        <pre className="checks-task-json">{JSON.stringify(task, null, 2)}</pre>
      </TechnicalDetails>
    </article>
  );
}

export function ResearchTaskTimeline({
  audit,
  tasks,
}: {
  audit: Audit;
  tasks: readonly ResearchTask[];
}) {
  return (
    <ol className="research-timeline">
      {tasks.map((task) => {
        const state = taskStatus(task);
        return (
          <li key={task.document.task_id}>
            <span className="research-timeline-mark" data-tone={state.tone}>
              <StatusGlyph tone={state.tone} />
            </span>
            <div className="research-timeline-copy">
              <h3>
                {task.hypothesis_snapshot?.statement ?? task.document.objective}
              </h3>
              <p className="ui-caption">
                {task.hypothesis_snapshot
                  ? "Hypothesis check"
                  : "Exploratory or supporting work"}{" "}
                · {task.attempts} Run attempts ·{" "}
                <RecordedTime value={task.updated_at} />
              </p>
              {task.gap_code ? (
                <p className="ui-caption">{readableValue(task.gap_code)}</p>
              ) : null}
              {task.result || task.findings || task.stop ? (
                <TaskResultView audit={audit} task={task} />
              ) : null}
            </div>
            <StatusChip tone={state.tone} size="sm">
              {state.label}
            </StatusChip>
          </li>
        );
      })}
    </ol>
  );
}

/** Keep the former endpoint workspace's list/detail pattern over current Tasks. */
export function ResearchWork({ audit }: { audit: Audit }) {
  const [params, setParams] = useSearchParams();
  const hypothesis = params.get("hypothesis") ?? undefined;
  const { query, pages } = useResearchTasks(audit, hypothesis);
  const selected = params.get("task");
  const filter = ["pending", "collected", "gap"].includes(
    params.get("work") ?? "",
  )
    ? params.get("work")!
    : "all";
  const visible = (query.data?.items ?? []).filter(
    (task) =>
      filter === "all" ||
      (filter === "gap"
        ? !!task.gap_code
        : filter === "pending"
          ? task.state !== "settled"
          : task.state === "settled" && !task.gap_code),
  );
  const selectedRead = useResearchTask(
    audit,
    selected,
    query.data !== undefined &&
      !query.data.items.some((task) => task.document.task_id === selected),
  );
  const chosen =
    (query.data?.items ?? []).find((t) => t.document.task_id === selected) ??
    (selected ? selectedRead.data : visible[0]);
  function choose(id: string) {
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        next.set("task", id);
        return next;
      },
      { preventScrollReset: true },
    );
  }
  const navigation = useListNavigation({
    count: visible.length,
    index: visible.findIndex(
      (t) => t.document.task_id === chosen?.document.task_id,
    ),
    onMove: (index) => {
      const task = visible[index];
      if (task) choose(task.document.task_id);
    },
    onOpen: (index) => {
      const task = visible[index];
      if (task) choose(task.document.task_id);
    },
  });
  return (
    <QueryView query={query} loading={<p role="status">Loading tasks…</p>}>
      {(data) => (
        <>
          <div className="research-filter">
            <AddResearchWork audit={audit} />
            <label>
              Work on this page
              <select
                className="ui-control"
                value={filter}
                onChange={(e) =>
                  setParams((previous) => {
                    const next = new URLSearchParams(previous);
                    next.set("work", e.target.value);
                    next.delete("task");
                    return next;
                  })
                }
              >
                <option value="all">All tasks</option>
                <option value="pending">Pending work</option>
                <option value="collected">Collected results</option>
                <option value="gap">Technical gaps</option>
              </select>
            </label>
            <p className="ui-caption">
              {data.items.length} tasks loaded
              {data.page.hasMore ? " · more pages available" : ""}. Filters
              describe this page.
            </p>
          </div>
          {hypothesis ? (
            <div className="ui-toolbar">
              <span className="ui-caption">Checks of one exact hypothesis</span>
              <Button
                size="sm"
                onClick={() =>
                  setParams((previous) => {
                    const next = new URLSearchParams(previous);
                    next.delete("hypothesis");
                    next.delete("historyCursor");
                    next.delete("task");
                    return next;
                  })
                }
              >
                Show all checks
              </Button>
            </div>
          ) : null}
          <div className="research-workspace">
            <div className="research-work-list">
              <ul {...navigation.containerProps}>
                {visible.map((task) => {
                  const state = taskStatus(task);
                  return (
                    <li key={task.document.task_id}>
                      <Button
                        variant="ghost"
                        className="research-work-choice"
                        aria-current={
                          chosen?.document.task_id === task.document.task_id
                            ? true
                            : undefined
                        }
                        aria-pressed={
                          chosen?.document.task_id === task.document.task_id
                        }
                        onClick={() => choose(task.document.task_id)}
                      >
                        <StatusGlyph tone={state.tone} />
                        <span>
                          <strong>
                            {task.hypothesis_snapshot?.statement ??
                              task.document.objective}
                          </strong>
                          <small>
                            {state.label} · {task.attempts} Run attempts
                          </small>
                        </span>
                      </Button>
                    </li>
                  );
                })}
              </ul>
              <p className="ui-caption">
                <Kbd>J</Kbd> <Kbd>K</Kbd> move · Enter to inspect
              </p>
              <CursorControls
                label="Task pages"
                {...pages.controls(data.page)}
              />
            </div>
            {chosen &&
            (!hypothesis || chosen.document.hypothesis?.id === hypothesis) ? (
              <ResearchTaskDetail
                key={chosen.document.task_id}
                audit={audit}
                task={chosen}
              />
            ) : selected ? (
              <QueryView query={selectedRead}>
                {() => <p>This task is outside the selected hypothesis.</p>}
              </QueryView>
            ) : (
              <p>No matching tasks on this page.</p>
            )}
          </div>
        </>
      )}
    </QueryView>
  );
}
