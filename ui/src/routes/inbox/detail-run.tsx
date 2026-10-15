import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { usePublicAPI } from "../../api/context";
import type { RunStatus } from "../../api/runs";
import { ContextLink } from "../../app/context-navigation";
import { formatElapsed } from "../../app/format";
import { QueryView } from "../../app/query-view";
import { RecordedTime } from "../../app/recorded-time";
import { runRecoveryReason, runStateLabel } from "../../app/vocabulary";
import { useSession } from "../../auth/session";
import { DetailPane, Facts, IdChip, TechnicalDetails } from "../../ui";
import { artifactDetailPath } from "../artifacts/paths";
import { recoveryProgress, useRetryModelConnection } from "../runs/recovery";
import { runStatusQuery } from "../runs/status";
import { deriveRunTriage } from "../runs/triage";
import type { InboxData } from "./data";
import {
  Block,
  DetailBar,
  Head,
  OpenLink,
  Recorded,
  type DetailContext,
} from "./detail-parts";
import { refKey, type InboxRowOf } from "./model";
import { primaryOutput, projectPath, runPath, sectionTitle } from "./present";

function RunCause({ run }: { run: RunStatus }) {
  const triage = deriveRunTriage(run);
  const issue = triage.issue;
  if (run.state !== "failed" && run.state !== "cancelled") return null;
  return (
    <Block
      title={
        issue?.source === "cancellation" ? "Why it stopped" : "Primary cause"
      }
    >
      <p className="inbox-cause">
        {issue === undefined
          ? "No failure reason was reported. Open the Run to inspect the latest attempt."
          : issue.message}
      </p>
      {issue?.participant === undefined ? null : (
        <p className="ui-caption">
          Reported by {issue.participant}
          {issue.logicalAgent === undefined ? "" : ` ${issue.logicalAgent}`}
        </p>
      )}
    </Block>
  );
}

/**
 * Model recovery of a waiting Run, with the Run page's Retry model
 * connection and its confirmation.
 */
function RunRecovery({ run }: { run: RunStatus }) {
  const [requested, setRequested] = useState(false);
  const retry = useRetryModelConnection(run, {
    onRequested: () => setRequested(true),
  });
  const recovery = run.recovery;
  if (recovery === undefined && !requested) return null;
  return (
    <Block title="Model connection">
      {recovery === undefined ? null : (
        <>
          <p className="inbox-cause">{runRecoveryReason(recovery.code)}</p>
          <p className="ui-caption">
            {recoveryProgress(recovery)} Completed work is retained.
          </p>
        </>
      )}
      {retry.control === null ? null : (
        <div className="inbox-actions">{retry.control}</div>
      )}
      {requested ? (
        <p className="ui-caption" role="status">
          Retry requested. The Run continues once the model answers.
        </p>
      ) : null}
      {retry.overlay}
    </Block>
  );
}

function RunPrimaryOutput({ run, data }: { run: RunStatus; data: InboxData }) {
  const primary = primaryOutput(run, run, false, data.workflowOutputs);
  return (
    <Block title="Primary result">
      {primary.state === "present" ? (
        <p>
          <ContextLink
            returnLabel="Inbox"
            to={artifactDetailPath(
              { kind: "run", id: run.runId },
              primary.artifact,
            )}
          >
            Open {primary.slot}
          </ContextLink>
          , the output this Workflow marks as its primary result.
        </p>
      ) : (
        <p>
          {primary.state === "missing"
            ? `The primary result ${primary.slot} was not produced. Open the Run to see every output.`
            : primary.state === "none"
              ? "This Workflow declares no primary result. Open the Run to see every output."
              : primary.state === "loading"
                ? "Reading the Workflow's declared outputs…"
                : "The declared outputs could not be read. Open the Run to see every output."}
        </p>
      )}
      <p className="ui-caption">
        A finished Run alone does not establish the quality of its result.
      </p>
    </Block>
  );
}

function LoadedRun({
  run,
  context,
  canOperate,
}: {
  run: RunStatus;
  context: DetailContext;
  canOperate: boolean;
}) {
  const project =
    run.projectId === undefined
      ? undefined
      : context.data.projectNames.get(run.projectId);
  const triage = deriveRunTriage(run);
  const state = runStateLabel(run.state);
  return (
    <>
      <Head
        tone={state.tone}
        state={state.label}
        title={run.workflow}
        meta={
          <>
            <IdChip value={run.runId} label="Run ID" />
            {run.projectId === undefined ? null : (
              <Link to={projectPath(run.projectId)}>
                {project ?? "Project"}
              </Link>
            )}
            {run.finishedAt === undefined ? null : (
              <span>
                Ended <RecordedTime value={run.finishedAt} />
              </span>
            )}
          </>
        }
      />
      <RunCause run={run} />
      <RunRecovery run={run} />
      {run.state === "succeeded" ? (
        // A finished Run's next step is its result.
        <RunPrimaryOutput run={run} data={context.data} />
      ) : (
        <Block title="Next step">
          <p>
            <strong>{triage.guidance.title}.</strong> {triage.guidance.message}
          </p>
          {triage.guidance.operationsPath === undefined ||
          !canOperate ? null : (
            <p>
              <Link to={triage.guidance.operationsPath}>
                Open Operations diagnostics
              </Link>
            </p>
          )}
          {run.state === "failed" ? (
            <p className="ui-caption">
              Continue from failed stage and Configure another Run are separate
              actions on the{" "}
              <ContextLink returnLabel="Inbox" to={runPath(run.runId)}>
                Run page
              </ContextLink>
              .
            </p>
          ) : null}
        </Block>
      )}
      <TechnicalDetails description="Stage, attempts and the cause's code.">
        <Facts
          className="inbox-facts"
          size="sm"
          items={[
            { term: "Stage", value: triage.stage ?? "—" },
            { term: "Attempts", value: triage.attemptCount },
            { term: "Duration", value: formatElapsed(triage.durationMs) },
            {
              term: "Cause code",
              value:
                triage.issue === undefined ? undefined : (
                  <code>{triage.issue.code}</code>
                ),
            },
            {
              term: "Retryable",
              value:
                triage.issue === undefined
                  ? undefined
                  : triage.issue.retryable === undefined
                    ? "Unknown"
                    : triage.issue.retryable
                      ? "Yes"
                      : "No",
            },
          ]}
        />
      </TechnicalDetails>
    </>
  );
}

/**
 * A Run of the Inbox (waiting for the model, failed or finished), read with
 * the Run page's own status read and polling.
 */
export function RunDetail({
  runId,
  row,
  context,
}: {
  runId: string;
  row: InboxRowOf<"run"> | undefined;
  context: DetailContext;
}) {
  const api = usePublicAPI();
  const { session } = useSession();
  const canOperate =
    session?.principal.capabilities.includes("operations") === true;
  const query = useQuery(runStatusQuery(api, runId));
  const label =
    row?.reason === "model"
      ? "Waiting for the model"
      : row?.reason === "failed"
        ? "Failed Run"
        : row?.reason === "finished"
          ? "Finished Run"
          : "Run";
  return (
    <DetailPane
      header={
        <DetailBar
          where={sectionTitle(row, "Runs")}
          label={label}
          actions={<OpenLink to={runPath(runId)}>Open Run</OpenLink>}
          context={context}
          focus={context.focusKey === refKey({ kind: "run", runId })}
        />
      }
    >
      <Recorded text={context.recorded} />
      <QueryView
        query={query}
        errorContext="The Run could not be loaded."
        loading={
          <p className="ui-caption" role="status">
            Loading the Run…
          </p>
        }
      >
        {(run) => (
          <LoadedRun run={run} context={context} canOperate={canOperate} />
        )}
      </QueryView>
    </DetailPane>
  );
}
