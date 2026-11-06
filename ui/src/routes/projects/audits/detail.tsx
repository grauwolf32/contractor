import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { readAuditArtifact } from "../../../api/audit-artifacts";
import {
  AUDIT_ID_PATTERN,
  auditPollInterval,
  getAudit,
  type Audit,
} from "../../../api/audits";
import { usePublicAPI } from "../../../api/context";
import { invalidateCrossProject } from "../../../api/cross-project";
import {
  invalidAPIResponse,
  PublicAPIError,
  requireData,
} from "../../../api/error";
import type { components } from "../../../api/generated/public";
import { PROJECT_ID_PATTERN } from "../../../api/projects";
import { queryKeys } from "../../../api/query-keys";
import { CursorControls } from "../../../app/cursor-controls";
import { ErrorNotice } from "../../../app/error-notice";
import { useURLCursorStack } from "../../../app/pagination";
import { pollInterval } from "../../../app/polling";
import { QueryView } from "../../../app/query-view";
import { checkStateLabel } from "../../../app/vocabulary";
import { Button, Facts, StatusChip, TechnicalDetails } from "../../../ui";
import "./checks.css";
import { taskResultPreview } from "./task-result";
import { AuditHypotheses } from "./hypotheses";
import { AuditControls } from "./controls";
import { AuditActivity } from "./activity";
import { auditProfileLabel } from "./labels";
import "./styles.css";

type Task = components["schemas"]["AuditTask"];
const TASK_STATES: Record<Task["state"], string> = {
  ready: "Accepted",
  selected: "Run selected",
  settled: "Collected",
};

function TaskResult({ auditId, task }: { auditId: string; task: Task }) {
  const api = usePublicAPI();
  const [open, setOpen] = useState(false);
  const result = task.result;
  const query = useQuery({
    queryKey: [
      "audit-task-result",
      auditId,
      task.document.task_id,
      result?.ref.revision,
    ],
    enabled: open && result !== undefined,
    queryFn: async ({ signal }) => {
      if (result === undefined) throw new Error("No saved result");
      const read = await readAuditArtifact(
        api,
        auditId,
        `task/${task.document.task_id}/result`,
        result.ref.revision,
        undefined,
        signal,
      );
      return {
        blob: read.blob,
        preview:
          read.mediaType === "application/json"
            ? read.blob.size <= 1048576
              ? taskResultPreview(await read.blob.text(), task.document.task_id)
              : `Saved JSON material (${read.blob.size} bytes). Download the complete Run result to inspect its exact per-task conclusions.`
            : read.mediaType.startsWith("text/")
              ? await read.blob.slice(0, 32768).text()
              : `Saved ${read.mediaType} material (${read.blob.size} bytes).`,
      };
    },
  });
  if (result === undefined) return <p>No collected result yet.</p>;
  return (
    <div className="ui-stack">
      <Button size="sm" onClick={() => setOpen(!open)}>
        {open ? "Hide saved result" : "Read saved result"}
      </Button>
      {open ? (
        <QueryView
          query={query}
          loading={<p role="status">Loading saved result…</p>}
        >
          {(data) => (
            <>
              <pre className="checks-task-json">{data.preview}</pre>
              {data.blob.size > 32768 ? (
                <p>
                  Preview limited to 32 KiB. Download the complete result below.
                </p>
              ) : null}
              <Button
                size="sm"
                onClick={() => {
                  const url = URL.createObjectURL(data.blob);
                  const link = document.createElement("a");
                  link.href = url;
                  link.download = `${task.document.task_id}.result`;
                  link.click();
                  setTimeout(() => URL.revokeObjectURL(url), 1000);
                }}
              >
                Download complete Run result
              </Button>
            </>
          )}
        </QueryView>
      ) : null}
    </div>
  );
}

function CommonAudit({ audit }: { audit: Audit }) {
  const api = usePublicAPI();
  const changing = auditPollInterval([audit]) !== false;
  const [params, setParams] = useSearchParams();
  const hypothesis = params.get("hypothesis") ?? undefined;
  const pages = useURLCursorStack({ param: "tasksCursor" });
  const cursor = pages.cursor;
  const tasks = useQuery({
    queryKey: [
      "audit-tasks",
      audit.auditId,
      audit.revision,
      cursor,
      hypothesis,
    ],
    queryFn: async ({ signal }) => {
      const page = requireData(
        await api.request((client) =>
          client.GET("/v1/audits/{auditId}/tasks", {
            signal,
            params: {
              path: { auditId: audit.auditId },
              query: {
                limit: 50,
                ...(cursor === undefined ? {} : { cursor }),
                ...(hypothesis === undefined
                  ? {}
                  : { hypothesis_id: hypothesis }),
              },
            },
          }),
        ),
      );
      if (
        page.page.hasMore &&
        (page.items.length === 0 ||
          page.page.nextCursor === undefined ||
          page.page.nextCursor === cursor)
      )
        throw invalidAPIResponse(200, "Invalid Task continuation");
      return page;
    },
    refetchInterval: pollInterval<components["schemas"]["AuditTaskPage"]>({
      interval: 2000,
      settled: () => !changing,
    }),
  });
  const state = checkStateLabel(audit.state);
  return (
    <article className="ui-stack">
      <header className="checks-section-heading">
        <h1>{auditProfileLabel(audit)}</h1>
        <StatusChip tone={state.tone}>{state.label}</StatusChip>
      </header>
      <AuditControls audit={audit} projectName={undefined} />
      {audit.stopReason === undefined ? null : (
        <p role="status">{audit.stopReason.message}</p>
      )}
      <Facts
        items={[
          { term: "Accepted tasks", value: audit.taskCount },
          { term: "Submitted runs", value: audit.submittedRunCount },
          { term: "Outstanding runs", value: audit.outstandingRunCount },
        ]}
      />
      <p>
        Each task uses the selected materials. A saved result can contain
        observations and limitations without creating a possible issue.
      </p>
      <Link to={`/projects/${encodeURIComponent(audit.projectId)}/issues`}>
        Project possible issues
      </Link>
      <AuditHypotheses audit={audit} />
      <AuditActivity audit={audit} />
      {hypothesis === undefined ? null : (
        <Button
          size="sm"
          onClick={() => {
            const next = new URLSearchParams(params);
            next.delete("hypothesis");
            next.delete("tasksCursor");
            setParams(next);
          }}
        >
          Show all checks
        </Button>
      )}
      <QueryView query={tasks} loading={<p role="status">Loading tasks…</p>}>
        {(data) => (
          <section className="ui-stack" aria-label="Tasks">
            {data.items.length === 0 ? <p>No tasks accepted yet.</p> : null}
            {data.items.map((task) => (
              <article
                className="checks-task ui-stack"
                key={task.document.task_id}
              >
                <h2>
                  {task.hypothesis_snapshot?.statement ??
                    task.document.objective}
                </h2>
                {task.hypothesis_snapshot?.scope.conditions ===
                undefined ? null : (
                  <p>{task.hypothesis_snapshot.scope.conditions}</p>
                )}
                <p>
                  {TASK_STATES[task.state]} · {task.attempts} Run attempts
                </p>
                {task.gap_code === undefined ? null : (
                  <p role="status">Collection gap: {task.gap_code}</p>
                )}
                <TaskResult auditId={audit.auditId} task={task} />
                <TechnicalDetails description="Task identity and exact selected materials.">
                  <pre className="checks-task-json">
                    {JSON.stringify(task, null, 2)}
                  </pre>
                </TechnicalDetails>
              </article>
            ))}
            <CursorControls label="Task pages" {...pages.controls(data.page)} />
          </section>
        )}
      </QueryView>
    </article>
  );
}

/** Minimal common Audit view; task attempts use ordinary Runs and retained outputs. */
export function ProjectAuditDetailRoute() {
  const { projectId = "", auditId = "" } = useParams();
  const api = usePublicAPI();
  const valid =
    PROJECT_ID_PATTERN.test(projectId) && AUDIT_ID_PATTERN.test(auditId);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const audit = useQuery({
    queryKey: queryKeys.audits.detail(auditId),
    enabled: valid,
    queryFn: ({ signal }) => getAudit(api, auditId, signal),
    refetchInterval: pollInterval<Audit>({
      interval: 1000,
      settled: (current) => auditPollInterval([current]) === false,
    }),
  });
  useEffect(() => {
    if (audit.error instanceof PublicAPIError && audit.error.status === 404) {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.projects.audits.all(projectId),
      });
      void invalidateCrossProject(queryClient);
      void navigate(`/projects/${encodeURIComponent(projectId)}/audits`, {
        replace: true,
      });
    }
  }, [audit.error, queryClient, navigate, projectId]);
  if (!valid)
    return <ErrorNotice error={new Error("This check address is invalid.")} />;
  return (
    <QueryView query={audit} loading={<p role="status">Loading check…</p>}>
      {(value) =>
        value.projectId === projectId ? (
          <CommonAudit key={value.auditId} audit={value} />
        ) : (
          <ErrorNotice
            error={new Error("This check does not belong to this project.")}
          />
        )
      }
    </QueryView>
  );
}
